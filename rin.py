import os
import asyncio
import datetime
import collections
import discord
import aiohttp
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

GUILD_ID = os.getenv("GUILD_ID")  # optional: instant sync to one server
guild_obj = discord.Object(id=int(GUILD_ID)) if GUILD_ID else None

LOG_WEBHOOK_URL = os.getenv("LOG_WEBHOOK_URL")  # webhook in your private #rin-logs channel
OWNER_ID = int(os.getenv("OWNER_ID", "0") or 0)

DM_DAILY_LIMIT = int(os.getenv("DM_DAILY_LIMIT", "0") or 0)         # per-user, in DMs (0 = no cap)
GUILD_DAILY_LIMIT = int(os.getenv("GUILD_DAILY_LIMIT", "0") or 0)   # per-guild cap/day (0 = no cap)

# --- Discord client ---
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# --- OpenAI client ---
client_oa = OpenAI(api_key=OPENAI_API_KEY)

# --- Memory ---
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
    # keep system + last 12 messages
    convos[user_id] = convos[user_id][:1] + convos[user_id][-12:]

# --- Usage counters (reset daily, in-memory) ---
usage_dm = collections.defaultdict(int)       # key: user_id
usage_guild = collections.defaultdict(int)    # key: guild_id
usage_day = datetime.date.today()

def _reset_if_new_day():
    global usage_day, usage_dm, usage_guild
    today = datetime.date.today()
    if today != usage_day:
        usage_day = today
        usage_dm.clear()
        usage_guild.clear()

def who_scope(interaction: discord.Interaction) -> tuple[str, int]:
    return ("dm", interaction.user.id) if interaction.guild is None else ("guild", interaction.guild.id)

def over_quota(scope: str, key: int) -> bool:
    _reset_if_new_day()
    if scope == "dm" and DM_DAILY_LIMIT:
        return usage_dm[key] >= DM_DAILY_LIMIT
    if scope == "guild" and GUILD_DAILY_LIMIT:
        return usage_guild[key] >= GUILD_DAILY_LIMIT
    return False

def bump_quota(scope: str, key: int):
    if scope == "dm":
        usage_dm[key] += 1
    else:
        usage_guild[key] += 1

# --- Webhook logging ---
async def log_event(title: str, fields: dict[str, str]):
    if not LOG_WEBHOOK_URL:
        return
    payload = {
        "embeds": [{
            "title": title,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "fields": [{"name": k, "value": v, "inline": False} for k, v in fields.items()]
        }]
    }
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(LOG_WEBHOOK_URL, json=payload)
    except Exception as e:
        print("[log webhook] failed:", e)

# --- OpenAI call with friendly errors ---
async def chat_openai(user_id: int, user_msg: str, model: str = "gpt-4o-mini"):
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
    except RateLimitError:
        return ("⚠️ The bot’s AI quota is currently exhausted or rate-limited.\n"
                "Please try again later (or top up billing).")
    except APIStatusError as e:
        msg = getattr(e, "message", None) or "Something went wrong. Please try again soon."
        return f"⚠️ OpenAI error {e.status}: {msg}"
    except (APIConnectionError, APIError):
        return "⚠️ I couldn’t reach the AI service. Please try again in a bit."
    except Exception as e:
        return f"⚠️ Unexpected error: {type(e).__name__}. Please try again."

def chunk(s: str, n: int = 1900):
    return [s[i:i+n] for i in range(0, len(s), n)]

# ----------------- Commands -----------------

@tree.command(name="setpersona", description="Set the bot's persona just for you.")
@app_commands.describe(persona="Describe the persona/tone you want.")
async def setpersona(interaction: discord.Interaction, persona: str):
    ensure_thread(interaction.user.id, persona)
    convos[interaction.user.id][0] = {"role": "system", "content": persona}
    await interaction.response.send_message("✅ Persona updated for your chats.")
    await log_event("🧠 Persona Updated", {
        "Where": "DM" if interaction.guild is None else f"{interaction.guild.name} ({interaction.guild_id})",
        "User": f"{interaction.user} ({interaction.user.id})",
        "Persona (first 180 chars)": (persona[:180] + "…") if len(persona) > 180 else persona
    })

@tree.command(name="reset", description="Clear your chat history with the bot.")
async def reset(interaction: discord.Interaction):
    convos.pop(interaction.user.id, None)
    await interaction.response.send_message("🧹 History cleared. Fresh start!")
    await log_event("♻️ History Reset", {
        "Where": "DM" if interaction.guild is None else f"{interaction.guild.name} ({interaction.guild_id})",
        "User": f"{interaction.user} ({interaction.user.id})"
    })

@tree.command(name="chat", description="Chat with the bot (uses your persona & history).")
@app_commands.describe(message="What do you want to say?")
async def chat(interaction: discord.Interaction, message: str):
    scope, key = who_scope(interaction)
    if over_quota(scope, key):
        limit = DM_DAILY_LIMIT if scope == "dm" else GUILD_DAILY_LIMIT
        return await interaction.response.send_message(
            f"⚠️ Daily {scope.upper()} limit reached ({limit}). Please try again tomorrow.",
            ephemeral=True
        )

    await interaction.response.defer()
    reply = await chat_openai(interaction.user.id, message)

    bump_quota(scope, key)
    where = "DM" if interaction.guild is None else f"{interaction.guild.name} ({interaction.guild_id})"
    await log_event("💬 Chat Command", {
        "Where": where,
        "User": f"{interaction.user} ({interaction.user.id})",
        "Prompt": (message[:300] + "…") if len(message) > 300 else message,
        "Reply chars": str(len(reply))
    })

    for part in chunk(reply):
        await interaction.followup.send(part)

@tree.command(name="ai_health", description="Check AI connectivity.")
async def ai_health(interaction: discord.Interaction):
    ok = "✅" if OPENAI_API_KEY else "❌"
    await interaction.response.send_message(f"API key: {ok} | Model: gpt-4o-mini")

@tree.command(name="guilds", description="Owner-only: list servers Rin is in.")
async def guilds_cmd(interaction: discord.Interaction):
    if interaction.user.id != OWNER_ID:
        return await interaction.response.send_message("Not authorized.", ephemeral=True)
    names = [f"{g.name} ({g.id})" for g in client.guilds]
    text = "• " + "\n• ".join(names) if names else "No servers."
    await interaction.response.send_message(text, ephemeral=True)

# ----------------- Server add/remove tracking -----------------

@client.event
async def on_guild_join(guild: discord.Guild):
    owner = f"{guild.owner} ({guild.owner_id})" if guild.owner_id else "unknown"
    await log_event("✅ Added to Server", {
        "Guild": f"{guild.name} ({guild.id})",
        "Owner": owner,
        "Members": str(getattr(guild, "member_count", "n/a"))
    })

@client.event
async def on_guild_remove(guild: discord.Guild):
    await log_event("❌ Removed from Server", {
        "Guild": f"{guild.name} ({guild.id})"
    })

# ----------------- Sync on ready -----------------

@client.event
async def on_ready():
    try:
        if guild_obj:
            tree.copy_global_to(guild=guild_obj)
            g_cmds = await tree.sync(guild=guild_obj)
            print(f"[READY] Instant guild sync -> {len(g_cmds)} cmds in {GUILD_ID}")
        cmds = await tree.sync()
        print(f"[READY] Global sync -> {len(cmds)} cmds as {client.user}")
    except Exception as e:
        print("[ERROR] Slash command sync failed:", e)

client.run(DISCORD_TOKEN)
