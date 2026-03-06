const systemGrid = document.getElementById("system-grid");
const modelSelect = document.getElementById("model-key");
const taskSelect = document.getElementById("task-select");
const customModelField = document.getElementById("custom-model-field");
const enableDiarization = document.getElementById("enable-diarization");
const diarizationFields = document.getElementById("diarization-fields");
const form = document.getElementById("transcription-form");
const submitButton = document.getElementById("submit-button");
const formNote = document.getElementById("form-note");
const sslBadge = document.getElementById("ssl-badge");
const hfTokenInput = form.querySelector('input[name="hf_token"]');
const jobStatus = document.getElementById("job-status");
const downloads = document.getElementById("downloads");
const transcriptPreview = document.getElementById("transcript-preview");
const helpDialog = document.getElementById("help-dialog");
const helpDialogTitle = document.getElementById("help-dialog-title");
const helpDialogBody = document.getElementById("help-dialog-body");
const helpDialogClose = document.getElementById("help-dialog-close");

const DEFAULT_MODEL_KEY = "distil-large-v3";
const FALLBACK_MODEL_KEY = "turbo";

let currentJobId = null;
let pollHandle = null;
let systemInfo = null;

const HELP_CONTENT = {
  file: {
    title: "Audio or video file",
    body: "Choose the recording you want to transcribe. MP3, WAV, M4A, MP4, and similar formats work well.",
  },
  model: {
    title: "Model",
    body: "This controls the speech-to-text engine. Distil-Whisper Large v3 is the default because it is fast and works well for English meetings. Use Whisper Large v3 when you need translation or stronger multilingual support.",
  },
  custom_model: {
    title: "Custom model id or local path",
    body: "Use this only if you already know you want a different Whisper-compatible model. Most users should leave it alone.",
  },
  task: {
    title: "Task",
    body: "Transcribe keeps the original spoken language. Translate converts supported speech into English text. Translation usually needs Whisper Large v3 rather than the English-only distil model.",
  },
  language: {
    title: "Language",
    body: "This tells the model what language to expect. For English meetings, leave it as en. Setting the language can improve accuracy and speed compared with auto-detection.",
  },
  beam_size: {
    title: "Beam size",
    body: "Beam size is how many possible wordings the model compares before choosing the final text. Higher values can help with difficult audio, but they make the job slower. Leave it at 5 unless the transcript quality is poor.",
  },
  temperature: {
    title: "Temperature",
    body: "Temperature controls how adventurous the model is when picking words. Low values are more stable and predictable. For transcripts, 0.0 is usually best. Increase it only if the model gets stuck or repeats itself on messy audio.",
  },
  initial_prompt: {
    title: "Initial prompt",
    body: "This is extra context you give the model before it starts. It is useful for company names, product names, acronyms, attendee names, or domain-specific jargon that you want spelled correctly.",
  },
  vad_filter: {
    title: "Use VAD to trim silence",
    body: "VAD means voice activity detection. It tries to skip silence, background noise, and non-speech parts before transcription. It usually makes meeting transcripts cleaner and faster.",
  },
  enable_diarization: {
    title: "Enable speaker diarization",
    body: "Diarization tries to split the recording into speaker turns like SPEAKER_00 and SPEAKER_01. Turn this on for meetings with multiple people talking.",
  },
  word_timestamps: {
    title: "Include word timestamps",
    body: "This stores timings for individual words instead of only whole segments. It is useful for fine-grained alignment, but it can add extra processing and larger output files.",
  },
  num_speakers: {
    title: "Exact speakers",
    body: "Use this when you know exactly how many people are speaking in the meeting. If you know the count, diarization is usually more reliable.",
  },
  min_speakers: {
    title: "Minimum speakers",
    body: "Use this when you do not know the exact number of speakers but you know the lower bound. Example: if at least two people are definitely speaking, set this to 2.",
  },
  max_speakers: {
    title: "Maximum speakers",
    body: "Use this when you do not know the exact number of speakers but you know the upper bound. Giving a reasonable range can improve diarization compared with leaving it fully open.",
  },
  diarization_exclusive: {
    title: "Use exclusive diarization",
    body: "Exclusive diarization forces one speaker label at a time for a cleaner, easier-to-read transcript. It is usually the best choice for meeting notes, even though real conversations can overlap.",
  },
};

async function loadSystemInfo() {
  const response = await fetch("/api/system");
  const data = await response.json();
  systemInfo = data;

  modelSelect.innerHTML = data.models
    .map(
      (model) =>
        `<option value="${model.key}">${model.label} - ${model.recommended_for}</option>`,
    )
    .join("");
  setDefaultModel();

  renderSystemGrid(data);
  sslBadge.textContent =
    data.ssl.provider === "rbc_security"
      ? "RBC SSL"
      : data.ssl.provider === "certifi"
        ? "Local CA bundle"
        : "System CA bundle";
}

function setDefaultModel() {
  const preferredKey = [...modelSelect.options].some((option) => option.value === DEFAULT_MODEL_KEY)
    ? DEFAULT_MODEL_KEY
    : FALLBACK_MODEL_KEY;
  modelSelect.value = preferredKey;
  customModelField.hidden = modelSelect.value !== "custom";
}

function ensureTranslationCompatibleModel() {
  if (taskSelect.value !== "translate") {
    return;
  }

  if (!["large-v3", "custom"].includes(modelSelect.value) && [...modelSelect.options].some((option) => option.value === "large-v3")) {
    modelSelect.value = "large-v3";
  }
  customModelField.hidden = modelSelect.value !== "custom";
}

function renderSystemGrid(data) {
  const runtime = data.runtime;
  let diarizationStatus = "Unavailable";
  if (data.diarization.available) {
    if (data.diarization.local_model_ready) {
      diarizationStatus =
        data.diarization.local_model_source === "bundled" ? "Bundled local model" : "Cached local model";
    } else if (data.diarization.hf_token_configured) {
      diarizationStatus = "Ready to download";
    } else {
      diarizationStatus = "Token needed for first download";
    }
  }

  const items = [
    ["OS", `${runtime.os_name} ${runtime.machine}`],
    ["Python", runtime.python_version],
    ["Accelerator", runtime.accelerator],
    ["Inference", `${runtime.device} / ${runtime.compute_type}`],
    ["SSL provider", data.ssl.provider],
    ["SSL detail", data.ssl.detail || "n/a"],
    ["Diarization", diarizationStatus],
    ["ffmpeg", data.diarization.ffmpeg_available ? "Present" : "Missing"],
  ];

  systemGrid.innerHTML = items
    .map(
      ([label, value]) => `
        <article class="system-card">
          <span>${label}</span>
          <strong>${value}</strong>
        </article>
      `,
    )
    .join("");
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "Submitting..." : "Run transcription";
}

function openHelpDialog(helpKey) {
  const help = HELP_CONTENT[helpKey];
  if (!help) {
    return;
  }

  if (typeof helpDialog.showModal !== "function") {
    window.alert(`${help.title}\n\n${help.body}`);
    return;
  }

  helpDialogTitle.textContent = help.title;
  helpDialogBody.textContent = help.body;
  helpDialog.showModal();
}

function renderJob(job) {
  const progressPercent = resolveProgressPercent(job);
  const etaLabel = estimateEtaLabel(job, progressPercent);

  jobStatus.className = `job-status ${job.status}`;
  jobStatus.innerHTML = `
    <div class="status-line">
      <strong>${job.status.toUpperCase()}</strong>
      <span>${job.status_message}</span>
    </div>
    <div class="progress-stack">
      <div class="progress-caption">
        <span>${progressPercent}% complete</span>
        ${etaLabel ? `<span>${etaLabel}</span>` : ""}
      </div>
      <div class="progress-track">
        <div class="progress-fill" style="width: ${progressPercent}%"></div>
      </div>
    </div>
    <div class="meta-line">
      <span>File: ${job.original_filename}</span>
      <span>Updated: ${new Date(job.updated_at).toLocaleString()}</span>
      ${
        job.metadata?.num_speakers_detected
          ? `<span>Transcript speakers: ${job.metadata.num_speakers_detected}</span>`
          : ""
      }
    </div>
    ${
      job.error
        ? `<div class="error-block">${job.error}</div>`
        : ""
    }
  `;

  downloads.innerHTML = "";
  transcriptPreview.textContent = job.transcript_preview || "";

  if (job.downloads) {
    Object.entries(job.downloads).forEach(([format, url]) => {
      const anchor = document.createElement("a");
      anchor.className = "download-pill";
      anchor.href = url;
      anchor.textContent = `Download ${format.toUpperCase()}`;
      downloads.appendChild(anchor);
    });
  }
}

function resolveProgressPercent(job) {
  const rawValue = Number(job.progress_percent);
  if (Number.isFinite(rawValue)) {
    return Math.max(0, Math.min(100, Math.round(rawValue)));
  }
  if (job.status === "completed") {
    return 100;
  }
  return 0;
}

function estimateEtaLabel(job, progressPercent) {
  if (job.status !== "running" || !job.started_at || progressPercent < 8 || progressPercent >= 100) {
    return "";
  }

  const elapsedMs = Date.now() - new Date(job.started_at).getTime();
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) {
    return "";
  }

  const remainingMs = (elapsedMs * (100 - progressPercent)) / progressPercent;
  if (!Number.isFinite(remainingMs) || remainingMs <= 0) {
    return "";
  }

  return `ETA ${formatRemainingTime(remainingMs)}`;
}

function formatRemainingTime(milliseconds) {
  const totalMinutes = Math.max(1, Math.round(milliseconds / 60000));
  if (totalMinutes < 60) {
    return `${totalMinutes} min`;
  }

  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  if (minutes === 0) {
    return `${hours} hr`;
  }
  return `${hours} hr ${minutes} min`;
}

async function pollJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const job = await response.json();
  renderJob(job);

  if (job.status === "completed" || job.status === "failed") {
    clearInterval(pollHandle);
    pollHandle = null;
  }
}

modelSelect.addEventListener("change", () => {
  customModelField.hidden = modelSelect.value !== "custom";
  ensureTranslationCompatibleModel();
});

taskSelect.addEventListener("change", () => {
  ensureTranslationCompatibleModel();
});

enableDiarization.addEventListener("change", () => {
  diarizationFields.hidden = !enableDiarization.checked;
});

document.querySelectorAll(".help-button").forEach((button) => {
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    openHelpDialog(button.dataset.helpKey);
  });
});

helpDialogClose.addEventListener("click", () => {
  helpDialog.close();
});

helpDialog.addEventListener("click", (event) => {
  const rect = helpDialog.getBoundingClientRect();
  const insideDialog =
    rect.top <= event.clientY &&
    event.clientY <= rect.bottom &&
    rect.left <= event.clientX &&
    event.clientX <= rect.right;

  if (!insideDialog) {
    helpDialog.close();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true);
  formNote.textContent = "Uploading file and queueing transcription job...";

  const formData = new FormData(form);

  try {
    if (formData.has("enable_diarization")) {
      const existingToken = `${formData.get("hf_token") || ""}`.trim();
      const localModelReady = Boolean(systemInfo?.diarization?.local_model_ready);

      if (!localModelReady && !existingToken) {
        const promptedToken = window.prompt(
          "Speaker diarization needs a one-time Hugging Face download. Enter a token with access to pyannote/speaker-diarization-community-1.",
        );
        const normalizedToken = (promptedToken || "").trim();
        if (!normalizedToken) {
          throw new Error(
            "A Hugging Face token is required the first time diarization downloads the model.",
          );
        }
        formData.set("hf_token", normalizedToken);
        hfTokenInput.value = normalizedToken;
      }

      formNote.textContent = localModelReady
        ? "Uploading file and queueing transcription job..."
        : "Uploading file and starting the one-time diarization model download...";
    }

    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Job submission failed.");
    }

    currentJobId = payload.job_id;
    renderJob(payload);
    formNote.textContent = "Job queued. Polling for updates...";

    if (pollHandle) {
      clearInterval(pollHandle);
    }
    pollHandle = setInterval(() => pollJob(currentJobId), 2000);
  } catch (error) {
    jobStatus.className = "job-status failed";
    jobStatus.textContent = error.message;
    formNote.textContent = "Submission failed. Check the settings and try again.";
  } finally {
    setBusy(false);
  }
});

diarizationFields.hidden = !enableDiarization.checked;
loadSystemInfo().catch((error) => {
  jobStatus.className = "job-status failed";
  jobStatus.textContent = `Failed to load system info: ${error.message}`;
});
