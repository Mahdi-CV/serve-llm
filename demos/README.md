# Demo folders

Live demo setups for serving models on AMD Instinct GPUs with Claude Code.

| Folder | Model | Server |
|--------|-------|--------|
| `instinct-demo/` | Qwen3.5-27B-FP8 | MI300X at 134.199.193.61 |
| `gpt-oss-demo/` | gpt-oss 20B / 120B | MI350X at 134.199.199.97 (do-mi350-node) |

Server-side scripts (`stage_demo.sh`, `check_docker_log.sh`, etc.) are already
deployed on both boxes. Nothing to set up on the servers.

---

## Setting up on a new laptop

### 1. SSH config

Add both servers to `~/.ssh/config`:

```
Host do-mi350-node
    HostName 134.199.199.97
    User root

Host mi300x-demo
    HostName 134.199.193.61
    User root
```

### 2. Skill (global install)

```bash
git clone git@github.com:Mahdi-CV/serve-llm.git \
  ~/.claude/skills/serving-llms-on-instinct
```

Claude Code auto-discovers skills in `~/.claude/skills/` — no further config needed.

### 3. Demo folders

```bash
git clone git@github.com:Mahdi-CV/serve-llm.git /tmp/serve-llm

cp -r /tmp/serve-llm/demos/gpt-oss-demo  ~/gpt-oss-demo
cp -r /tmp/serve-llm/demos/instinct-demo ~/instinct-demo
```

---

## Per-take flow

### gpt-oss demo (MI350X)

```bash
cd ~/gpt-oss-demo && ./stage.sh
claude --dangerously-skip-permissions
```

Say:
> run gpt-oss 120B on my digital ocean server with vLLM and give me the endpoint

After the endpoint is up:
```bash
ssh do-mi350-node
./set_hermes_custom_model.sh
hermes chat
```

### Qwen3.5 demo (MI300X)

```bash
cd ~/instinct-demo && ./stage.sh
claude --dangerously-skip-permissions
```

Say:
> run Qwen3.5 27B on my Instinct server at 134.199.193.61 with vLLM and give me the endpoint

---

## Optional: live server view

Open a second terminal before the Claude run:

```bash
# tmux split (log + process + health):
./watch.sh

# or just the raw log:
ssh do-mi350-node   # or mi300x-demo
./check_docker_log.sh
```

---

## Keeping the skill up to date

```bash
cd ~/.claude/skills/serving-llms-on-instinct && git pull origin main
```
