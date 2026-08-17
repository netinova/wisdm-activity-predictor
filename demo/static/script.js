/*
 *
 * CONFIGURATION
 *
 * SENSOR_HZ = how frequently we sample the sensor
 *
 * WINDOW_SECONDS = how long one prediction window is
 *
 * Example:
 *
 * 20 Hz × 4 seconds = 80 samples
 *
 *
 */

const API_URL = "http://localhost:8000/predict";

const SENSOR_HZ = 20;

const WINDOW_SECONDS = 4;

const SAMPLE_INTERVAL_MS = 1000 / SENSOR_HZ;

const WINDOW_SIZE = SENSOR_HZ * WINDOW_SECONDS;

// state

let isTracking = false;

let currentMode = "pocket";

let sensorBuffer = [];

let totalSamples = 0;

let sampleTimer = null;

let lastSensorReading = null;

let receivedMotionEvent = false;

let receivedAcceleration = false;

let receivedGyroscope = false;

let sensorWatchdogTimer = null;

// dom

const startButton = document.getElementById("startButton");

const stopButton = document.getElementById("stopButton");

const pocketButton = document.getElementById("pocketButton");

const handButton = document.getElementById("handButton");

const predictionText = document.getElementById("predictionText");

const confidenceText = document.getElementById("confidenceText");

const predictionIcon = document.getElementById("predictionIcon");

const predictionTime = document.getElementById("predictionTime");

const statusDot = document.getElementById("statusDot");

const statusText = document.getElementById("statusText");

const motionApiStatus = document.getElementById("motionApiStatus");

const motionApiIndicator = document.getElementById("motionApiIndicator");

const motionEventStatus = document.getElementById("motionEventStatus");

const motionEventIndicator = document.getElementById("motionEventIndicator");

const accelStatus = document.getElementById("accelStatus");

const accelIndicator = document.getElementById("accelIndicator");

const gyroStatus = document.getElementById("gyroStatus");

const gyroIndicator = document.getElementById("gyroIndicator");

const secureConnectionBadge = document.getElementById("secureConnectionBadge");

const sampleCount = document.getElementById("sampleCount");

const currentModeText = document.getElementById("currentMode");

const windowProgress = document.getElementById("windowProgress");

const windowTotal = document.getElementById("windowTotal");

const windowProgressBar = document.getElementById("windowProgressBar");

const sensorWarning = document.getElementById("sensorWarning");

const sensorWarningText = document.getElementById("sensorWarningText");

const logContainer = document.getElementById("logContainer");

const clearLogButton = document.getElementById("clearLogButton");

// initializing

windowTotal.textContent = WINDOW_SIZE;

document.getElementById("windowSampleCount").textContent = WINDOW_SIZE;

document.getElementById("pipelineRate").textContent = `${SENSOR_HZ} Hz`;

document.getElementById("pipelineWindow").textContent = `${WINDOW_SECONDS} sec`;

function getTime() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function addLog(title, message, type = "normal") {
  const entry = document.createElement("div");

  entry.className = "log-entry";

  let indicatorColor = "";

  if (type === "error") {
    indicatorColor = 'style="background:#ef4444"';
  }

  if (type === "warning") {
    indicatorColor = 'style="background:#f59e0b"';
  }

  entry.innerHTML = `
        <span
            class="log-indicator"
            ${indicatorColor}
        ></span>

        <div>
            <strong>${title}</strong>

            <p>${message}</p>
        </div>

        <time>${getTime()}</time>
    `;

  logContainer.prepend(entry);
}

function checkSensorEnvironment() {
  const motionAvailable = typeof DeviceMotionEvent !== "undefined";

  if (motionAvailable) {
    motionApiStatus.textContent = "Available";

    motionApiIndicator.classList.add("active");
  } else {
    motionApiStatus.textContent = "Unavailable";

    motionApiIndicator.classList.add("error");

    showSensorWarning("This browser does not expose DeviceMotionEvent.");

    addLog(
      "Motion API unavailable",
      "The browser does not support device motion sensors.",
      "error",
    );
  }

  if (window.isSecureContext) {
    secureConnectionBadge.textContent = "Secure context";

    secureConnectionBadge.classList.add("good");

    addLog(
      "Secure context",
      "The page is running in a secure browser context.",
    );
  } else {
    secureConnectionBadge.textContent = "Not secure";

    secureConnectionBadge.classList.add("bad");

    showSensorWarning(
      "Device sensors may be blocked because this page is not running in a secure context. Use HTTPS.",
    );

    addLog("Insecure connection", "Sensor APIs may require HTTPS.", "warning");
  }

  if (
    motionAvailable &&
    typeof DeviceMotionEvent.requestPermission === "function"
  ) {
    addLog(
      "Permission required",
      "This device requires permission before motion sensors can be used.",
    );
  } else if (motionAvailable) {
    addLog(
      "Sensor permission",
      "No explicit motion permission request is required by this browser.",
    );
  }
}

function showSensorWarning(message) {
  sensorWarning.classList.remove("hidden");

  sensorWarningText.textContent = message;
}

function hideSensorWarning() {
  sensorWarning.classList.add("hidden");
}

function handleMotion(event) {
  if (!isTracking) {
    return;
  }

  receivedMotionEvent = true;

  motionEventStatus.textContent = "Receiving";

  motionEventIndicator.classList.add("active");

  const acceleration = event.accelerationIncludingGravity;

  const rotation = event.rotationRate;

  if (acceleration) {
    receivedAcceleration = true;

    accelStatus.textContent = "Receiving";

    accelIndicator.classList.add("active");
  }

  if (rotation) {
    receivedGyroscope = true;

    gyroStatus.textContent = "Receiving";

    gyroIndicator.classList.add("active");
  }

  if (!acceleration) {
    return;
  }

  lastSensorReading = {
    x_accel: acceleration.x ?? 0,

    y_accel: acceleration.y ?? 0,

    z_accel: acceleration.z ?? 0,

    x_gyro: rotation?.alpha ?? 0,

    y_gyro: rotation?.beta ?? 0,

    z_gyro: rotation?.gamma ?? 0,
  };
}

/*
 *
 * Every 50 ms we create one sample.
 *
 * 20 samples
 *     ↓
 * 1 second
 *
 * 80 samples
 *     ↓
 * 4 seconds
 *
 *     ↓
 *
 * POST to API
 */

function collectSample() {
  if (!isTracking) {
    return;
  }
  if (!lastSensorReading) {
    updateWindowProgress();

    return;
  }

  const sample = {
    timeStamp: Date.now(),

    x_accel: lastSensorReading.x_accel,

    y_accel: lastSensorReading.y_accel,

    z_accel: lastSensorReading.z_accel,

    x_gyro: lastSensorReading.x_gyro,

    y_gyro: lastSensorReading.y_gyro,

    z_gyro: lastSensorReading.z_gyro,
  };

  sensorBuffer.push(sample);

  totalSamples++;

  sampleCount.textContent = totalSamples;

  updateWindowProgress();

  if (sensorBuffer.length >= WINDOW_SIZE) {
    const window = sensorBuffer.slice(0, WINDOW_SIZE);

    sensorBuffer.splice(0, WINDOW_SIZE);

    updateWindowProgress();

    sendSensorWindow(window);
  }
}

function updateWindowProgress() {
  const count = Math.min(sensorBuffer.length, WINDOW_SIZE);

  windowProgress.textContent = count;

  const percentage = (count / WINDOW_SIZE) * 100;

  windowProgressBar.style.width = `${percentage}%`;
}

async function startTracking() {
  if (isTracking) {
    return;
  }

  if (typeof DeviceMotionEvent === "undefined") {
    addLog(
      "Cannot start",
      "DeviceMotionEvent is not available in this browser.",
      "error",
    );

    return;
  }

  if (typeof DeviceMotionEvent.requestPermission === "function") {
    addLog(
      "Requesting sensor permission",
      "Waiting for permission from the device...",
    );

    try {
      const permission = await DeviceMotionEvent.requestPermission();

      if (permission !== "granted") {
        addLog(
          "Permission denied",
          "Motion sensor permission was not granted.",
          "error",
        );

        showSensorWarning("Motion sensor permission was denied.");

        return;
      }

      addLog("Permission granted", "Motion sensors are now accessible.");
    } catch (error) {
      console.error(error);

      addLog("Permission error", error.message, "error");

      return;
    }
  }

  isTracking = true;

  sensorBuffer = [];

  totalSamples = 0;

  lastSensorReading = null;

  receivedMotionEvent = false;

  receivedAcceleration = false;

  receivedGyroscope = false;

  sampleCount.textContent = "0";

  updateWindowProgress();

  motionEventStatus.textContent = "Waiting";

  accelStatus.textContent = "Waiting";

  gyroStatus.textContent = "Waiting";

  motionEventIndicator.classList.remove("active", "error");

  accelIndicator.classList.remove("active", "error");

  gyroIndicator.classList.remove("active", "error");

  hideSensorWarning();

  startButton.disabled = true;

  stopButton.disabled = false;

  pocketButton.disabled = true;

  handButton.disabled = true;

  statusDot.classList.add("active");

  statusText.textContent = "Tracking";

  predictionText.textContent = "Collecting data...";

  predictionIcon.textContent = "◌";

  confidenceText.textContent = `Building ${WINDOW_SECONDS}-second window`;

  window.addEventListener("devicemotion", handleMotion);

  sampleTimer = setInterval(collectSample, SAMPLE_INTERVAL_MS);

  sensorWatchdogTimer = setTimeout(checkSensorReception, 2000);

  addLog(
    "Tracking started",
    `${SENSOR_HZ} Hz sampling · ${WINDOW_SECONDS}s windows · ${WINDOW_SIZE} samples/prediction`,
  );
}

function checkSensorReception() {
  if (!isTracking) {
    return;
  }

  if (!receivedMotionEvent) {
    motionEventStatus.textContent = "No events";

    motionEventIndicator.classList.add("error");

    showSensorWarning(
      "No DeviceMotion events have been received. Make sure you're using a phone, grant sensor permission, and serve this page over HTTPS.",
    );

    addLog(
      "No sensor events",
      "The browser has not delivered any DeviceMotion events.",
      "error",
    );

    return;
  }

  if (!receivedAcceleration) {
    showSensorWarning(
      "Motion events are arriving, but accelerometer data is unavailable.",
    );

    addLog(
      "Accelerometer unavailable",
      "Motion events exist but acceleration data is missing.",
      "error",
    );

    return;
  }

  if (!receivedGyroscope) {
    showSensorWarning(
      "Motion events are arriving, but gyroscope data is unavailable.",
    );

    addLog(
      "Gyroscope unavailable",
      "Motion events exist but gyroscope data is missing.",
      "warning",
    );

    return;
  }

  hideSensorWarning();

  addLog(
    "Sensors operational",
    "Accelerometer and gyroscope data are being received.",
  );
}

async function stopTracking() {
  if (!isTracking) {
    return;
  }

  isTracking = false;

  window.removeEventListener("devicemotion", handleMotion);

  if (sampleTimer !== null) {
    clearInterval(sampleTimer);

    sampleTimer = null;
  }

  if (sensorWatchdogTimer !== null) {
    clearTimeout(sensorWatchdogTimer);

    sensorWatchdogTimer = null;
  }

  const remainingSamples = sensorBuffer.length;

  sensorBuffer = [];

  startButton.disabled = false;

  stopButton.disabled = true;

  pocketButton.disabled = false;

  handButton.disabled = false;

  statusDot.classList.remove("active");

  statusText.textContent = "Idle";

  accelStatus.textContent = "Inactive";

  gyroStatus.textContent = "Inactive";

  motionEventStatus.textContent = "Stopped";

  predictionIcon.textContent = "—";

  addLog(
    "Tracking stopped",
    `Collected ${totalSamples} samples. Discarded ${remainingSamples} incomplete window samples.`,
  );
}

async function sendSensorWindow(windowData) {
  if (windowData.length !== WINDOW_SIZE) {
    addLog(
      "Window rejected",
      `Expected ${WINDOW_SIZE} samples, got ${windowData.length}.`,
      "error",
    );

    return;
  }

  addLog(
    "Sending prediction",
    `${windowData.length} samples → ${currentMode} model`,
  );

  try {
    /*
     * Mode is sent as a query parameter.
     *
     * POST /predict?mode=pocket
     *
     * Body:
     *
     * {
     *     data: [...]
     * }
     */

    const url = `${API_URL}?mode=${encodeURIComponent(currentMode)}`;

    const response = await fetch(url, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        data: windowData,
      }),
    });

    if (!response.ok) {
      throw new Error(`API returned HTTP ${response.status}`);
    }

    const result = await response.json();

    const prediction =
      result.prediction ?? result.activity ?? result.label ?? "Unknown";

    const confidence = result.confidence;

    updatePrediction(prediction, confidence);

    addLog("Prediction received", `${formatPrediction(prediction)}`);
  } catch (error) {
    console.error("Prediction request failed:", error);

    addLog("API error", error.message, "error");
  }
}

function updatePrediction(prediction, confidence) {
  predictionText.textContent = formatPrediction(prediction);

  if (typeof confidence === "number") {
    confidenceText.textContent = `${(confidence * 100).toFixed(1)}% confidence`;
  } else {
    confidenceText.textContent = "Prediction received from model";
  }

  predictionIcon.textContent = getActivityIcon(prediction);

  predictionTime.textContent = `Updated ${getTime()}`;
}

function formatPrediction(value) {
  return String(value)
    .replace(/[_-]/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

const ACTIVITY_ICONS = {
  Walking: "🚶",
  Jogging: "🏃",
  Stairs: "🪜",
  Sitting: "🪑",
  Standing: "🧍",
  Typing: "⌨️",
  "Brushing Teeth": "🪥",
  "Eating Soup": "🥣",
  "Eating Chips": "🥔",
  "Eating Pasta": "🍝",
  "Drinking from Cup": "☕",
  "Eating Sandwich": "🥪",
  "Kicking (Soccer Ball)": "⚽",
  "Playing Catch w/Tennis Ball": "🎾",
  "Dribbling (Basketball)": "🏀",
  Writing: "✍️",
  Clapping: "👏",
  "Folding Clothes": "👕",
};

function getActivityIcon(activity) {
  if (ACTIVITY_ICONS[activity]) {
    return ACTIVITY_ICONS[activity];
  }

  const normalized = String(activity).trim().toLowerCase();

  const match = Object.entries(ACTIVITY_ICONS).find(
    ([name]) => name.toLowerCase() === normalized,
  );

  return match ? match[1] : "●";
}

function setMode(mode) {
  if (isTracking) {
    addLog(
      "Mode change blocked",
      "Stop tracking before changing the model.",
      "warning",
    );

    return;
  }

  currentMode = mode;

  pocketButton.classList.toggle("active", mode === "pocket");

  handButton.classList.toggle("active", mode === "hand");

  currentModeText.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);

  addLog(
    "Model mode changed",
    `Using ${mode} mode for the next prediction session.`,
  );
}

pocketButton.addEventListener("click", () => setMode("pocket"));

handButton.addEventListener("click", () => setMode("hand"));

clearLogButton.addEventListener("click", () => {
  logContainer.innerHTML = "";

  addLog("Log cleared", "Activity log has been cleared.");
});

startButton.addEventListener("click", startTracking);

stopButton.addEventListener("click", stopTracking);

checkSensorEnvironment();
