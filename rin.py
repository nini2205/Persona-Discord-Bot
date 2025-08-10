import os
import asyncio
import discord
from discord import app_commands, Embed
from openai import OpenAI
from openai import APIStatusError, RateLimitError, APIConnectionError, APIError

# --- Env ---
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

# Optional: instant slash-command sync to your server
GUILD_ID = os.getenv("GUILD_ID")
guild_obj = discord.Object(id=int(GUILD_ID)) if GUILD_ID else None

# --- Discord client ---
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# --- OpenAI client ---
client_oa = OpenAI(api_key=OPENAI_API_KEY)

# --- Simple per-user memory (in RAM) ---
# user_id -> list[{"role": "system|user|assistant", "content": "..."}]
convos = {}

DEFAULT_PERSONA = (
    "You are Rin, a kind and helpful swim club captain. "
    "Be helpful and upbeat, avoid walls of text, and format responses cleanly. "
    "If the user asks for code, provide runnable snippets."
)

def ensure_thread(user_id: int, persona: str | None = None):
    if user_id not in convos:
        convos[user_id] = [{"role": "system", "content": persona or DEFAULT_PERSONA}]
    # keep history short to control cost: keep system + last 12 msgs
    convos[user_id] = convos[user_id][:1] + convos[user_id][-12:]

async def chat_openai(user_id: int, user_msg: str, model: str = "gpt-4o-mini"):
    """Call OpenAI in a thread; return a friendly message on errors."""
    ensure_thread(user_id)
    convos[user_id].append({"role": "user", "content": user_msg})

    def _call():
        return client_oa.chat.completions.create(
            model=model,
            messages=convos[user_id],
            temperature=0.7,
        )

    try:
        resp = await asyncio.to_thread(_call)
        text = resp.choices[0].message.content or ""
        convos[user_id].append({"role": "assistant", "content": text})
        return text

    # ----- Friendly error handling -----
    except RateLimitError:
        return ("⚠️ The bot’s AI quota is currently exhausted or rate-limited.\n"
                "Please try again later (or top up billing).")
    except APIStatusError as e:
        # 4xx/5xx with JSON body
        msg = getattr(e, "message", None) or "Something went wrong. Please try again soon."
        return f"⚠️ OpenAI error {e.status}: {msg}"
    except (APIConnectionError, APIError):
        return "⚠️ I couldn’t reach the AI service. Please try again in a bit."
    except Exception as e:
        # Last-resort catch to keep bot alive
        return f"⚠️ Unexpected error: {type(e).__name__}. Please try again."

def chunk(s: str, n: int = 1900):
    # split long replies to respect Discord's 2000 char limit
    return [s[i:i+n] for i in range(0, len(s), n)]

# ----------------- Commands -----------------

@tree.command(name="setpersona", description="Set the bot's persona just for you.")
@app_commands.describe(persona="Describe the persona/tone you want.")
async def setpersona(interaction: discord.Interaction, persona: str):
    ensure_thread(interaction.user.id, persona)
    # replace system message
    convos[interaction.user.id][0] = {"role": "system", "content": persona}
    await interaction.response.send_message("✅ Persona updated for your chats.")

@tree.command(name="reset", description="Clear your chat history with the bot.")
async def reset(interaction: discord.Interaction):
    convos.pop(interaction.user.id, None)
    await interaction.response.send_message("🧹 History cleared. Fresh start!")

@tree.command(name="chat", description="Chat with the bot (uses your persona & history).")
@app_commands.describe(message="What do you want to say?")
async def chat(interaction: discord.Interaction, message: str):
    await interaction.response.defer()  # give us time
    reply = await chat_openai(interaction.user.id, message)
    for part in chunk(reply):
        await interaction.followup.send(part)

@tree.command(name="ai_health", description="Check AI connectivity.")
async def ai_health(interaction: discord.Interaction):
    ok = "✅" if OPENAI_API_KEY else "❌"
    await interaction.response.send_message(f"API key: {ok} | Model: gpt-4o-mini")

# Basic ping
@tree.command(name="ping", description="Latency test.")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message(f"Pong! {round(client.latency*1000)} ms")

# Sync on ready (guild for instant, else global)
@client.event
async def on_ready():
    try:
        if guild_obj:
            # 1) Instant for your test server
            tree.copy_global_to(guild=guild_obj)
            g_cmds = await tree.sync(guild=guild_obj)
            print(f"[READY] Instant guild sync -> {len(g_cmds)} cmds in {GUILD_ID}")

        # 2) Also push global in parallel (may take up to ~1 hour to appear)
        cmds = await tree.sync()
        print(f"[READY] Global sync -> {len(cmds)} cmds as {client.user}")

    except Exception as e:
        print("[ERROR] Slash command sync failed:", e)

client.run(DISCORD_TOKEN)
