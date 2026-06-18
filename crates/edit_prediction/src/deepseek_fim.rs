use crate::{
    EditPredictionId, EditPredictionModelInput, cursor_excerpt, prediction::EditPredictionResult,
};
use anyhow::{Context as _, Result};
use futures::AsyncReadExt as _;
use gpui::{
    App, AppContext as _, Entity, Global, SharedString, Task,
    http_client::{self, HttpClient},
};
use language::{Anchor, ToOffset, ToPoint as _, language_settings::all_language_settings};
use language_model::{ApiKeyState, AuthenticateError, EnvVar, env_var};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc, time::Instant};
use zeta_prompt::{ZetaPromptInput, compute_editable_and_context_ranges};

/// Default base URL for DeepSeek's Fill-in-the-Middle completion API.
///
/// FIM completion requires the beta endpoint, which is why this differs from
/// the `/v1` base URL used by the chat completion provider.
/// See <https://api-docs.deepseek.com/guides/fim_completion>.
pub const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/beta";

/// Number of tokens around the cursor used to build the FIM prefix and suffix.
const DEEPSEEK_FIM_CONTEXT_TOKENS: usize = 512;

const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-chat";
const DEFAULT_DEEPSEEK_MAX_TOKENS: u32 = 256;

static DEEPSEEK_API_KEY_ENV_VAR: std::sync::LazyLock<EnvVar> = env_var!("DEEPSEEK_API_KEY");

struct GlobalDeepSeekApiKey(Entity<ApiKeyState>);

impl Global for GlobalDeepSeekApiKey {}

pub fn deepseek_api_url(cx: &App) -> SharedString {
    all_language_settings(None, cx)
        .edit_predictions
        .deepseek
        .api_url
        .clone()
        .filter(|api_url| !api_url.is_empty())
        .unwrap_or_else(|| DEEPSEEK_API_URL.to_string())
        .into()
}

pub fn deepseek_api_key_state(cx: &mut App) -> Entity<ApiKeyState> {
    if let Some(global) = cx.try_global::<GlobalDeepSeekApiKey>() {
        return global.0.clone();
    }
    let entity =
        cx.new(|cx| ApiKeyState::new(deepseek_api_url(cx), DEEPSEEK_API_KEY_ENV_VAR.clone()));
    cx.set_global(GlobalDeepSeekApiKey(entity.clone()));
    entity
}

pub fn deepseek_api_key(cx: &App) -> Option<Arc<str>> {
    let url = deepseek_api_url(cx);
    cx.try_global::<GlobalDeepSeekApiKey>()?
        .0
        .read(cx)
        .key(&url)
}

pub fn load_deepseek_api_key(cx: &mut App) -> Task<Result<(), AuthenticateError>> {
    let credentials_provider = zed_credentials_provider::global(cx);
    let api_url = deepseek_api_url(cx);
    deepseek_api_key_state(cx).update(cx, |key_state, cx| {
        key_state.load_if_needed(api_url, |s| s, credentials_provider, cx)
    })
}

struct FimRequestOutput {
    request_id: String,
    edits: Vec<(std::ops::Range<Anchor>, Arc<str>)>,
    editable_range: std::ops::Range<Anchor>,
    snapshot: language::BufferSnapshot,
    inputs: ZetaPromptInput,
    buffer: Entity<language::Buffer>,
}

pub fn request_prediction(
    EditPredictionModelInput {
        buffer,
        snapshot,
        position,
        events,
        trigger,
        ..
    }: EditPredictionModelInput,
    cx: &mut App,
) -> Task<Result<Option<EditPredictionResult>>> {
    let settings = &all_language_settings(None, cx).edit_predictions.deepseek;
    let model = settings
        .model
        .clone()
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| DEFAULT_DEEPSEEK_MODEL.to_string());
    let max_tokens = settings.max_tokens.unwrap_or(DEFAULT_DEEPSEEK_MAX_TOKENS);

    // Ensure stored credentials are loaded in the background so subsequent
    // requests can pick them up, then read whatever key is currently available
    // (env var or already-loaded credentials).
    load_deepseek_api_key(cx).detach();
    let Some(api_key) = deepseek_api_key(cx) else {
        return Task::ready(Ok(None));
    };
    let api_url = deepseek_api_url(cx).to_string();

    let full_path: Arc<Path> = snapshot
        .file()
        .map(|file| file.full_path(cx))
        .unwrap_or_else(|| "untitled".into())
        .into();

    let http_client = cx.http_client();
    let cursor_point = position.to_point(&snapshot);
    let request_start = cx.background_executor().now();

    let result = cx.background_spawn(async move {
        let cursor_offset = cursor_point.to_offset(&snapshot);
        let (excerpt_point_range, excerpt_offset_range, cursor_offset_in_excerpt) =
            cursor_excerpt::compute_cursor_excerpt(&snapshot, cursor_offset);
        let cursor_excerpt: Arc<str> = snapshot
            .text_for_range(excerpt_point_range.clone())
            .collect::<String>()
            .into();
        let syntax_ranges =
            cursor_excerpt::compute_syntax_ranges(&snapshot, cursor_offset, &excerpt_offset_range);
        let (editable_range, _) = compute_editable_and_context_ranges(
            &cursor_excerpt,
            cursor_offset_in_excerpt,
            &syntax_ranges,
            DEEPSEEK_FIM_CONTEXT_TOKENS,
            0,
        );

        let inputs = ZetaPromptInput {
            events,
            related_files: Some(Vec::new()),
            active_buffer_diagnostics: Vec::new(),
            cursor_offset_in_excerpt: cursor_offset - excerpt_offset_range.start,
            cursor_path: full_path.clone(),
            excerpt_start_row: Some(excerpt_point_range.start.row),
            cursor_excerpt,
            excerpt_ranges: Default::default(),
            syntax_ranges: None,
            in_open_source_repo: false,
            can_collect_data: false,
            repo_url: None,
        };

        let editable_text = &inputs.cursor_excerpt[editable_range.clone()];
        let cursor_in_editable = cursor_offset_in_excerpt.saturating_sub(editable_range.start);
        let prefix = editable_text[..cursor_in_editable].to_string();
        let suffix = editable_text[cursor_in_editable..].to_string();

        let (response_text, request_id) =
            send_deepseek_fim_request(&http_client, &api_url, &api_key, model, prefix, suffix, max_tokens)
                .await?;

        log::debug!(
            "deepseek fim: completion received ({:.2}s)",
            (Instant::now() - request_start).as_secs_f64()
        );

        let completion: Arc<str> = response_text.into();
        let edits = if completion.is_empty() {
            vec![]
        } else {
            let anchor = snapshot.anchor_after(cursor_offset);
            vec![(anchor..anchor, completion)]
        };

        let editable_range = snapshot.anchor_range_inside(
            (excerpt_offset_range.start + editable_range.start)
                ..(excerpt_offset_range.start + editable_range.end),
        );

        anyhow::Ok(FimRequestOutput {
            request_id,
            edits,
            editable_range,
            snapshot,
            inputs,
            buffer,
        })
    });

    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
        let output = result.await.context("deepseek fim edit prediction failed")?;
        anyhow::Ok(Some(
            EditPredictionResult::new(
                EditPredictionId(output.request_id.into()),
                &output.buffer,
                &output.snapshot,
                output.edits.into(),
                None,
                Some(output.editable_range),
                output.inputs,
                None,
                trigger,
                cx.background_executor().now() - request_start,
                cx,
            )
            .await,
        ))
    })
}

#[derive(Serialize)]
struct DeepSeekFimRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<String>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Deserialize)]
struct DeepSeekFimResponse {
    #[serde(default)]
    id: String,
    choices: Vec<DeepSeekFimChoice>,
}

#[derive(Deserialize)]
struct DeepSeekFimChoice {
    text: String,
}

async fn send_deepseek_fim_request(
    http_client: &Arc<dyn HttpClient>,
    api_url: &str,
    api_key: &str,
    model: String,
    prefix: String,
    suffix: String,
    max_tokens: u32,
) -> Result<(String, String)> {
    let request = DeepSeekFimRequest {
        model,
        prompt: prefix,
        suffix: if suffix.is_empty() { None } else { Some(suffix) },
        max_tokens,
        temperature: 0.0,
    };

    let request_body = serde_json::to_string(&request)?;
    let http_request = http_client::Request::builder()
        .method(http_client::Method::POST)
        .uri(format!("{}/completions", api_url.trim_end_matches('/')))
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .body(http_client::AsyncBody::from(request_body))?;

    let mut response = http_client.send(http_request).await?;
    let status = response.status();

    let mut body = String::new();
    response.body_mut().read_to_string(&mut body).await?;

    if !status.is_success() {
        anyhow::bail!("DeepSeek FIM API error: {} - {}", status, body);
    }

    let parsed: DeepSeekFimResponse =
        serde_json::from_str(&body).context("Failed to parse DeepSeek FIM response")?;
    let text = parsed
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.text)
        .unwrap_or_default();
    Ok((text, parsed.id))
}
