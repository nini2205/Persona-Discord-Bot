import os
import asyncio
import discord
from discord import app_commands, Embed
from openai import OpenAI

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
    "You are Rio the Kitty Butler—warm, witty, concise. "
    "Be helpful and upbeat, avoid walls of text, and format responses cleanly. "
    "If the user asks for code, provide runnable snippets."
)

def ensure_thread(user_id: int, persona: str | None = None):
    if user_id not in convos:
        convos[user_id] = [{"role": "system", "content": persona or DEFAULT_PERSONA}]
    # keep history short to control cost
    # retain system + last 12 messages max
    convos[user_id] = convos[user_id][:1] + convos[user_id][-12:]

async def chat_openai(user_id: int, user_msg: str, model: str = "gpt-4o-mini"):
    """Call OpenAI in a thread to avoid blocking the event loop."""
    ensure_thread(user_id)
    convos[user_id].append({"role": "user", "content": user_msg})

    def _call():
        return client_oa.chat.completions.create(
            model=model,
            messages=convos[user_id],
            temperature=0.7,
        )

    resp = await asyncio.to_thread(_call)
    text = resp.choices[0].message.content or ""
    convos[user_id].append({"role": "assistant", "content": text})
    return text

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

# Basic ping
@tree.command(name="ping", description="Latency test.")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message(f"Pong! {round(client.latency*1000)} ms")

# Sync on ready (guild for instant, else global)
@client.event
async def on_ready():
    try:
        if guild_obj:
            tree.copy_global_to(guild=guild_obj)
            cmds = await tree.sync(guild=guild_obj)
            print(f"[READY] Synced {len(cmds)} cmds to guild {GUILD_ID} as {client.user}")
        else:
            cmds = await tree.sync()
            print(f"[READY] Globally synced {len(cmds)} cmds as {client.user}")
    except Exception as e:
        print("[ERROR] Slash command sync failed:", e)

client.run(DISCORD_TOKEN)
