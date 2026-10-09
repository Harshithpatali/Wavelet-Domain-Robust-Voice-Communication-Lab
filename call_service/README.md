# Private Live Call Signaling Service

This is the separate FastAPI/WebSocket signaling service for the Streamlit Wavelet Voice Lab. It relays WebRTC offers, answers, and ICE candidates only. The browsers exchange microphone media over WebRTC; this service does not receive or record the media stream.

## Deploy as a regular Render Docker Web Service

Use a normal **Web Service** rather than a Blueprint.

1. In Render, choose **New → Web Service** and connect the GitHub repository.
2. Select branch `main`.
3. Choose **Docker** as the runtime.
4. Set **Dockerfile Path** to `call_service/Dockerfile`.
5. Set **Docker Context Directory** to `.` (the repository root). The Dockerfile copies `call_service/requirements.txt` and the `call_service/` package from this root context.
6. Choose the nearest available region (Singapore is a reasonable starting point for users in India).
7. Set the health-check path to `/healthz`.
8. Use one instance and one Uvicorn worker. Live room socket state is process-local, so multiple workers/instances are not supported by this starter service.

The container automatically binds Uvicorn to Render's `PORT` environment variable.

### Required environment variables

- `CALL_CREATION_TOKEN`: set a long random secret (at least 32 characters). It authorizes room creation from the Streamlit server. Generate it in Render or another password generator and never expose it in browser code or email links.
- `ALLOWED_ORIGINS`: set to `https://wavelet-domain-robust-voice-communication-lab.streamlit.app`. For local development, append `http://localhost:8501` separated by a comma.
- `DATABASE_URL`: recommended. Use your existing Neon PostgreSQL connection string to persist expiring room metadata in `wavelet_call_rooms`. Only a salted PBKDF2 hash of the access code is stored. Without this variable, rooms exist only in the running process and are lost on restart or sleep.
- `ICE_SERVERS_JSON`: optional JSON array for custom STUN/TURN configuration. The default is a public STUN server. Some networks require TURN; for more reliable calling, configure a reputable TURN service and use short-lived credentials where supported.

After deployment, your service URL should be HTTPS, and its WebSocket signaling will use WSS. Verify `https://YOUR-SERVICE.onrender.com/healthz` returns `{"status":"ok"}` before configuring Streamlit.

> **Free-instance consideration:** a free service may sleep when idle, so the first invitation can be slow to create and an idle service may not be suitable for uninterrupted calling. Use a plan that stays active if you need reliable availability. This is a Render hosting constraint, not a code feature.

## Configure Streamlit Community Cloud

In the Streamlit app's Secrets settings, add:

```toml
CALL_SIGNALING_URL = "https://YOUR-SERVICE.onrender.com"
CALL_CREATION_TOKEN = "THE-SAME-SERVER-ONLY-SECRET-AS-RENDER"
```

The `CALL_CREATION_TOKEN` must exactly match the value set on the signaling service. The Streamlit Python process sends it to create rooms; it is never inserted into the invitation link or browser widget.

## Invite flow

1. Open **Private Live Call** in the sidebar and create a room with an expiry.
2. Email the invitation link to the other person.
3. Send the 8-digit access code through a separate channel where possible.
4. Both parties open the app link, enter the code, and press **Join call**.
5. Compare the peer-verification code shown on both screens using a trusted independent channel before sharing sensitive information.

The user interface includes mute/unmute, end call, connection state, a round-trip estimate, and the peer-verification code.

## What the live-call mode does—and does not do

- **Implemented as a research mode:** when the live wavelet checkbox is enabled (default), each browser applies a five-level Haar DWT to 960-sample/48 kHz blocks, keyed coefficient permutation and inverse DWT reconstruction before sending audio to the WebRTC codec. The receiving browser uses the matching key to reverse the coefficient permutation.
- Both callers must enter exactly the same separate wavelet code (minimum eight characters). It remains local to the browser and is not sent through signaling. This is a weak seeded permutation, not cryptographic encryption.
- WebRTC uses DTLS-SRTP to encrypt media in transit regardless of the wavelet checkbox. The recipient's browser decrypts that transport and, if wavelet mode is enabled, attempts the inverse signal processing before playback.
- The **invitation link** identifies the room, and the separate **8-digit access code** authorizes joining. The room code is not the wavelet code or media-encryption key; WebRTC negotiates transport keys automatically.
- The wavelet path is a prototype and may create artifacts because the WebRTC codec is lossy and block alignment matters. The unit tests validate the DSP transform offline; testing between separate real devices is still required.
- This browser app does not interface with RF walkie-talkies. Physical-radio integration would require compatible hardware or a radio gateway and a separately engineered/approved radio security system. Do not deploy the prototype for operational police communications.
- The signaling service relays setup messages, not audio. The room access code is stored as a salted PBKDF2 hash rather than plaintext.
- The peer-verification code is derived from the exchanged DTLS fingerprints. Compare it independently; a room code alone does not prove the other caller's real-world identity.
- This prototype has not had an independent security audit. Do not describe it as an audited end-to-end encrypted product.
- No camera or recording is implemented. The browser asks for microphone permission only after the caller presses Join.
- The default STUN server cannot guarantee connectivity on every network. TURN may be required.
- The prototype supports one-to-one rooms on a single signaling instance. Live sockets will disconnect during service restarts; horizontal scaling needs shared connection coordination.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\\.venv\\Scripts\\Activate.ps1
pip install -r call_service/requirements.txt
export CALL_CREATION_TOKEN="local-development-token-change-me-123456"
uvicorn call_service.app:app --reload --port 8000
```

In another terminal run `streamlit run app.py` and set `CALL_SIGNALING_URL=http://localhost:8000` plus `CALL_CREATION_TOKEN` in Streamlit secrets or environment variables. Only use HTTP for local development.

Run the tests from the repository root:

```bash
pip install -r requirements.txt -r call_service/requirements.txt
pytest -q
```
