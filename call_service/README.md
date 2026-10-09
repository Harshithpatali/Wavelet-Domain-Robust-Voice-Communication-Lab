# Private Live Call Signaling Service

This is the separate FastAPI/WebSocket signaling service for the Streamlit Wavelet Voice Lab. It relays WebRTC offers, answers, and ICE candidates only. The browsers exchange microphone media directly over WebRTC; the signaling service does not receive or record audio.

## Deploy on Render

Create a **Web Service** linked to this repository and branch \`main\` after the private-call pull request is merged.

- Runtime: Python
- Build command: \`pip install -r call_service/requirements.txt\`
- Start command: \`uvicorn call_service.app:app --host 0.0.0.0 --port $PORT\`
- Region: choose the closest available region
- Instances/workers: use one service instance and one Uvicorn worker because connected WebSocket rooms are process-local

Configure these Render environment variables:

- \`CALL_CREATION_TOKEN\`: generate a random secret with at least 32 characters. This server-to-server secret authorizes room creation and must not be exposed in the browser.
- \`DATABASE_URL\`: optional but recommended; use the existing Neon Postgres connection string. Persistent invitations are saved in \`wavelet_call_rooms\`, with only a salted PBKDF2 hash of the access code. The plaintext code is returned only when the invitation is created.
- \`ALLOWED_ORIGINS\`: comma-separated Streamlit origin(s), for example \`https://wavelet-domain-robust-voice-communication-lab.streamlit.app,http://localhost:8501\`.
- \`ICE_SERVERS_JSON\`: optional JSON array containing STUN/TURN server configuration. The default is a public STUN server. Production deployments should configure a reputable TURN provider with credentials that can be rotated or time-limited.

The service exposes \`/healthz\` for health checks and \`/api/ice\` to provide ICE server configuration. Use the Render HTTPS/WSS URL; do not use plain HTTP/WSS in production.

## Configure Streamlit Community Cloud

In the Streamlit app's Secrets settings, add:

\`\`\`toml
CALL_SIGNALING_URL = "https://YOUR-SERVICE.onrender.com"
CALL_CREATION_TOKEN = "THE-SAME-SECRET-AS-RENDER"
\`\`\`

The creation token is only used by the Streamlit Python server to create a room. It is not passed to the embedded browser widget. Do not put the token in a query string, email, or public repository.

## Invite flow

1. Open **Private Live Call** in the sidebar and create an invitation.
2. Share the generated app link with the recipient.
3. Share the 8-digit access code separately when possible.
4. Both parties open the link, enter the access code, and press **Join call**.
5. The host and guest negotiate audio with WebRTC. The UI includes mute, leave, connection state, latency estimate, and a peer-verification code derived from the exchanged DTLS fingerprints.

The peer-verification code should match on both screens. Compare it through a separate trusted channel before discussing sensitive information. Stop the call if codes do not match.

## Security and operating limits

- WebRTC media uses DTLS-SRTP encryption; the signaling service does not handle media packets.
- The access code protects room entry but is not, by itself, proof of a person's identity. The peer-verification code provides an additional manual check against signaling substitution; it is not a formal security audit.
- The default public STUN service cannot guarantee connectivity on every network. Some networks require TURN relay configuration.
- With \`DATABASE_URL\`, invitations persist across service restarts until expiry. Live socket connections themselves do not survive a restart.
- Without \`DATABASE_URL\`, invitations live only in process memory and disappear if the service restarts/sleeps.
- The starter service targets one-to-one calls and one service instance. Horizontal scaling requires shared room/connection coordination.
- The browser never requests a camera, does not record audio, and asks for microphone permission only when the user presses Join call.
- Do not present the wavelet-coefficient permutation as cryptographic protection. It remains an educational signal-processing mode separate from WebRTC media encryption.

## Local development

\`\`\`bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\\.venv\\Scripts\\Activate.ps1
pip install -r call_service/requirements.txt
export CALL_CREATION_TOKEN="local-development-token-change-me-123456"
uvicorn call_service.app:app --reload --port 8000
\`\`\`

In another terminal run \`streamlit run app.py\` and set \`CALL_SIGNALING_URL\` and \`CALL_CREATION_TOKEN\` in Streamlit secrets or environment variables.

Run the service tests from the repository root:

\`\`\`bash
pip install -r requirements.txt -r call_service/requirements.txt
pytest -q
\`\`\`
