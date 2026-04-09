const { spawn } = require('child_process');
const fs = require('fs');

const transcriptPath = '/Users/antoinepinelli/.cursor/projects/Users-antoinepinelli-Cursor-App-VoiceGenius/agent-transcripts/93ba20f1-d38d-4ef4-b2ec-e063ae9bb5a7/93ba20f1-d38d-4ef4-b2ec-e063ae9bb5a7.jsonl';
let content = '';
const lines = fs.readFileSync(transcriptPath, 'utf8').split('\n');
for (const line of lines) {
    if (!line.trim()) continue;
    try {
        const obj = JSON.parse(line);
        if (obj.role && obj.message && obj.message.content) {
            content += `${obj.role.toUpperCase()}: ${obj.message.content[0].text}\n\n`;
        }
    } catch(e) {}
}

const child = spawn('/Users/antoinepinelli/.local/bin/MemoryPilot', []);

let state = 0;
let request_id = 1;

child.stdout.on('data', (data) => {
    const lines = data.toString().split('\n');
    for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (state === 0 && msg.id === 1) {
            // Initialized
            state = 1;
            child.stdin.write(JSON.stringify({
                jsonrpc: "2.0",
                id: 2,
                method: "notifications/initialized",
                params: {}
            }) + "\n");
            
            // Send transcript
            child.stdin.write(JSON.stringify({
                jsonrpc: "2.0",
                id: 3,
                method: "tools/call",
                params: {
                    name: "add_transcript",
                    arguments: {
                        content: content,
                        project: "voicegenius",
                        tags: ["history", "voicegenius-v1"]
                    }
                }
            }) + "\n");
        } else if (state === 1 && msg.id === 3) {
            console.log("MCP Response:", msg.result.content[0].text);
            process.exit(0);
        }
    }
});

child.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
});

// Initialize
child.stdin.write(JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "test-client", version: "1.0" }
    }
}) + "\n");
