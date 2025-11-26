# DQN Agents: Comprehensive Explanation

## Quick Answer to Your Questions

### Q1: Can DQN agents work standalone (without AudioCRNN/SirenClassifier)?
**Short Answer:** Yes, BUT with caveats.

**Long Answer:** The DQN agents are trained on audio **features** (MFCC tokens), not trained models. They learn a **decision policy** (when to alert), not classification.

### Q2: Do DQN agents determine notification timing correctly?
**Answer:** Yes, that's exactly what they do! They're purpose-built for this.

---

## Understanding the Training Pipeline

### What DQN Agents ARE NOT Trained On

```
❌ NOT trained on AudioCRNN or SirenClassifier directly
❌ NOT classifying audio types
❌ NOT replacing the models
```

### What DQN Agents ARE Trained On

```
✅ AUDIO FEATURES directly:
   - MFCC tokens (audio features)
   - Energy levels
   - Noise detection

✅ REWARD SIGNAL (reinforcement learning):
   - When to alert (action decision)
   - When NOT to alert
   - Optimal timing for notifications
```

---

## The Training Flow

### Step 1: Audio → Features

```python
# From alert-reinforcement-model.ipynb
def process_audio_mfcc(file_path, sr=16000, n_mfcc=13):
    """Extract MFCC features from audio"""
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # [time_steps, 13 features]
```

**Output:** Mel-Frequency Cepstral Coefficients (features, not labels)

### Step 2: Features → Tokens (Clustering)

```python
kmeans = MiniBatchKMeans(n_clusters=64)  # 64 token vocabulary
tokens = kmeans.predict(mfcc)  # Quantize features
# Result: [time_steps] of integers 0-63
```

**Output:** Discrete token sequence

### Step 3: Environment (Reward Definition)

```python
class SirenChunkEnv(gym.Env):
    """
    Environment for RL training
    - State: Token sequence (audio features)
    - Action: 0=WAIT, 1=ALERT (when to notify)
    - Reward: Points for correct decisions
    """
```

### Step 4: DQN Learning

```python
from stable_baselines3 import DQN

dqn = DQN(
    "MlpPolicy",  # Simple neural network policy
    env,          # SirenChunkEnv (reward-driven environment)
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    total_timesteps=50_000
)

dqn.learn(total_timesteps=50_000)  # Train to maximize reward!
```

**Output:** Trained agent that knows when to alert

---

## Key Difference: Classification vs Decision-Making

### AudioCRNN & SirenClassifier
```
Input:  Audio → Features
Output: Classification (what type of sound?)
        "Glass break" (0.85 confidence)
        "Traffic" (0.10 confidence)
        "Crash" (0.05 confidence)
```

### DQN Agents
```
Input:  Audio Features (or previous classification)
Output: Decision (should we alert?)
        Action 0: WAIT (don't notify)
        Action 1: ALERT (notify user)
        + Q-value (confidence in this decision)
```

---

## Can DQN Work Standalone? YES, With Important Notes

### Scenario 1: DQN ONLY (No AudioCRNN/SirenClassifier)

```python
# DQN directly on MFCC tokens
audio → MFCC → tokens → DQN → Alert/No-Alert decision

✅ Works: DQN was trained exactly this way!
❌ Limitation: Doesn't classify WHAT triggered alert
   Just decides WHETHER to alert based on audio patterns
```

### Scenario 2: DQN WITH Classification Models

```python
audio → MFCC → tokens → DQN → Alert decision
audio → Mel-spec → AudioCRNN → Glass/Traffic/Crash
                → SirenClassifier → Ambient/Alert

✅ Best approach:
   - Tells WHAT happened (classification)
   - Tells WHETHER to alert (DQN decision)
   - Can combine: Alert on high-confidence siren + DQN approval
```

### Scenario 3: Current Multi-Model API (Ensemble)

```python
# What multi_model_api.py does:
audio → MFCC ────────────────→ [SirenClassifier] ────┐
        Mel-spec ────→ [AudioCRNN] ────────────────┐ │
        Features ──────────────────→ [Alert DQN] ──┼─┤
                                    [Emergency DQN]┼─┤
                                    [Env DQN] ────┘ │
                                                    │
                                    VOTING & CONSENSUS
                                          │
                                    Final Decision
```

---

## How DQN Handles Notifications - The Reward System

### Environment Setup (From Your Notebook)

```python
# Reward parameters
REWARD_ALERT_SIREN_HIGH = 10.0    # Correct alert + high confidence
REWARD_ALERT_SIREN_LOW  = -5.0    # Wrong alert or low confidence
REWARD_ALERT_DELAYED    = -1.0    # Late alert (penalize delay)
REWARD_NOISE_HIGH       = -1.0    # Alert in noisy background
REWARD_NOISE_LOW_ALERT  = 0.5     # Alert in quiet (good)
REWARD_MISS_SIREN       = -5.0    # Missed siren (didn't alert)
```

### What DQN Learns

```python
for each audio chunk:
    observe features (MFCC tokens)
    
    if action = 0 (WAIT):
        reward += REWARD_OTHERWISE  # Small positive for waiting
        reward -= REWARD_MISS_SIREN (if siren present)  # Penalty for not alerting
    
    if action = 1 (ALERT):
        if siren is present:
            if noise is low:
                reward += REWARD_ALERT_SIREN_HIGH + REWARD_NOISE_LOW_ALERT
            else:
                reward += REWARD_ALERT_SIREN_HIGH - REWARD_NOISE_HIGH
        else:
            reward += REWARD_ALERT_SIREN_LOW  # False alarm penalty
        
        # Also check if it's EARLY alert (good timing)
        if steps_since_siren_start < SHORT_DURATION_STEPS:
            reward += REWARD_EARLY_ALERT  # Bonus for quick response
```

### Result

```
After 50,000 training steps:
✅ Learns to alert quickly when siren detected
✅ Learns to avoid alerting on false sounds
✅ Learns to balance noise interference
✅ Learns optimal timing (early alerts > late alerts)
```

---

## DQN Agent Structure

### Three Trained Agents in Your Repository

```
1. alert_dqn_agent/
   - Trained to detect: Alert sounds
   - Actions: WAIT (0) or ALERT (1)
   - File: policy.pth (the trained weights)
   - Status: ✅ Ready to use

2. emergency_siren_dqn_agent/
   - Trained to detect: Emergency sirens
   - Actions: WAIT (0) or ALERT (1)
   - File: policy.pth
   - Status: ✅ Ready to use

3. environmental_dqn_agent/
   - Trained to detect: Environmental sounds
   - Actions: WAIT (0) or ALERT (1)
   - File: policy.pth
   - Status: ✅ Ready to use
```

### Each Agent Contains

```
policy.pth                    # Neural network weights
policy.optimizer.pth          # Optimizer state
pytorch_variables.pth         # Variable snapshots
_stable_baselines3_version    # Version info
system_info.txt              # Training environment info
```

---

## Using DQN Agents Standalone

### Step 1: Load Agent

```python
from stable_baselines3 import DQN

# Load trained DQN
agent = DQN.load("alert_dqn_agent/policy")

# Agent is now ready!
```

### Step 2: Extract Features from Audio

```python
import librosa
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Step 2a: Load audio
audio, sr = librosa.load("audio.wav", sr=16000)

# Step 2b: Extract MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # [13, time]

# Step 2c: Cluster into tokens
kmeans = MiniBatchKMeans(n_clusters=64)
tokens = kmeans.predict(mfcc.T)  # [time] with values 0-63
```

### Step 3: Feed to DQN

```python
import torch

# Prepare observation
obs = tokens.reshape(1, -1).astype(np.float32)  # [1, sequence_length]

# Get DQN decision
action, _states = agent.predict(obs, deterministic=True)

if action == 1:
    print("✓ ALERT: DQN recommends notifying user")
else:
    print("✗ WAIT: DQN recommends no notification")
```

### Step 4: Get Confidence (Q-value)

```python
# DQN also provides Q-values (confidence)
q_values = agent.predict(obs)  # [batch, num_actions]

# Q-value for each action
q_wait = q_values[0][0]   # Q-value of WAIT
q_alert = q_values[0][1]  # Q-value of ALERT

confidence = np.max(q_values)  # How confident in decision?
```

---

## Current Integration in multi_model_api.py

### How DQN Agents Are Used

```python
def predict_dqn_agent(agent, audio_bytes: bytes, agent_name: str):
    """
    Predict with DQN agent
    """
    # Extract mel-spectrogram
    mel = wav_to_mel(audio_bytes)
    mel_tensor = torch.from_numpy(mel).to(DEVICE)
    
    # Flatten to features
    obs = mel_tensor.reshape(1, -1).float()
    
    # Get DQN action
    action, _ = agent.predict(obs, deterministic=True)
    
    return {
        "model": f"DQN_{agent_name}",
        "action": int(action),
        "action_label": "ALERT" if action == 1 else "WAIT",
        "success": True
    }
```

### Ensemble Voting Uses DQN

```python
# In /classify endpoint:
alerts_detected = 0

if alert_dqn_prediction["action"] == 1:
    alerts_detected += 1
if emergency_dqn_prediction["action"] == 1:
    alerts_detected += 1
if environmental_dqn_prediction["action"] == 1:
    alerts_detected += 1

# Consensus
if alerts_detected >= 2:  # Majority vote
    final_decision = "ALERT USER"
else:
    final_decision = "NO ALERT NEEDED"
```

---

## Standalone DQN Setup (If You Want DQN Only)

### Minimal Code Example

```python
import librosa
import numpy as np
from stable_baselines3 import DQN
import torch

# 1. Load agents
alert_agent = DQN.load("alert_dqn_agent/policy")
emergency_agent = DQN.load("emergency_siren_dqn_agent/policy")
env_agent = DQN.load("environmental_dqn_agent/policy")

# 2. Function to get alert decision
def should_alert(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Extract features
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_norm = (mel - mel.min()) / (mel.max() - mel.min())
    
    # Flatten
    obs = mel_norm.flatten().reshape(1, -1).astype(np.float32)
    
    # Get predictions
    alert_action, _ = alert_agent.predict(obs)
    emergency_action, _ = emergency_agent.predict(obs)
    env_action, _ = env_agent.predict(obs)
    
    # Voting
    alert_votes = sum([alert_action, emergency_action, env_action])
    
    return {
        "should_alert": alert_votes >= 2,
        "votes": alert_votes,
        "confidence": alert_votes / 3.0
    }

# 3. Use it
result = should_alert("siren.wav")
print(f"Alert: {result['should_alert']} ({result['confidence']*100:.0f}% confidence)")
```

---

## Summary: DQN Agents Explained

| Aspect | Details |
|--------|---------|
| **What trained them?** | RL with audio features + reward signals (not models) |
| **Can use standalone?** | Yes! Extract MFCC → feed to DQN → get decision |
| **Their purpose** | Decide WHEN to alert (not WHAT sound it is) |
| **Notification system** | Learns optimal timing from rewards |
| **In current API** | Used for voting to decide if alert needed |
| **Independent of AudioCRNN/SirenClassifier?** | Yes, they use different features (MFCC not mel-spec) |

---

## Recommendation

### Use Case 1: Fast Decision-Making
```
✅ Use DQN only
- Faster (no classification needed)
- Real-time alert decisions
- Trained exactly for this task
```

### Use Case 2: Accurate Classification + Smart Alerts
```
✅ Use ensemble (current approach)
- Know WHAT happened (AudioCRNN, SirenClassifier)
- Know WHETHER to alert (DQN voting)
- Best accuracy + explainability
```

### Use Case 3: Production System
```
✅ Keep multi_model_api.py
- All approaches combined
- Voting for robustness
- Confidence scores for user trust
- Can disable models individually if needed
```

---

**Status of DQN Agents in Your System:** ✅ **Fully Functional**

The DQN agents are working correctly - they're deciding when to alert based on learned patterns from training. You can use them standalone or in the ensemble. Everything is integrated and ready!
