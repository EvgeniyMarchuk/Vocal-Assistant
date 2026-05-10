#!/usr/bin/env python3
import logging
import os
import re
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

try:
    from mutagen import File as MutagenFile
except Exception:  # pragma: no cover
    MutagenFile = None

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Установите TELEGRAM_BOT_TOKEN в переменных окружения")

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
FFMPEG_BIN = shutil.which("ffmpeg")


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "неизвестно"
    return f"{seconds:.2f} сек"


def extract_audio_info(file_path: Path) -> dict:
    info = {
        "duration_sec": None,
        "sample_rate_hz": None,
        "channels": None,
        "bitrate_bps": None,
        "codec": None,
    }

    if MutagenFile is not None:
        try:
            audio = MutagenFile(file_path)
            if audio is not None and getattr(audio, "info", None) is not None:
                meta = audio.info
                info["duration_sec"] = getattr(meta, "length", None)
                info["sample_rate_hz"] = getattr(meta, "sample_rate", None)
                info["channels"] = getattr(meta, "channels", None)
                info["bitrate_bps"] = getattr(meta, "bitrate", None)
                info["codec"] = meta.__class__.__name__
        except Exception:
            logger.exception("Mutagen failed to parse %s", file_path)

    # Fallback для WAV, если mutagen не дал нужные поля
    if file_path.suffix.lower() == ".wav" and (
        info["duration_sec"] is None or info["sample_rate_hz"] is None
    ):
        try:
            with wave.open(str(file_path), "rb") as wav:
                sr = wav.getframerate()
                frames = wav.getnframes()
                channels = wav.getnchannels()
                info["sample_rate_hz"] = info["sample_rate_hz"] or sr
                info["channels"] = info["channels"] or channels
                if sr > 0:
                    info["duration_sec"] = info["duration_sec"] or (frames / sr)
                info["codec"] = info["codec"] or "WAV"
        except Exception:
            logger.exception("wave fallback failed for %s", file_path)

    return info


def resolve_extension(file_path: Optional[str], fallback: str = ".bin") -> str:
    if not file_path:
        return fallback
    suffix = Path(file_path).suffix.lower()
    return suffix if suffix else fallback


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    if FFMPEG_BIN is None:
        raise RuntimeError("ffmpeg не найден в PATH")

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "Привет! Пришли voice/audio, я сконвертирую его в WAV, сохраню и верну метаданные."
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "Я принимаю:\n"
        "- голосовые (voice)\n"
        "- аудиофайлы (audio)\n\n"
        "После отправки конвертирую файл в WAV, сохраню в bot/storage/<chat_id>/ и верну метаданные."
    )


async def audio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return

    media = msg.voice or msg.audio
    if media is None:
        return

    chat_id = str(msg.chat_id)
    chat_dir = STORAGE_DIR / safe_name(chat_id)
    chat_dir.mkdir(parents=True, exist_ok=True)

    file_id = media.file_id
    telegram_file = await context.bot.get_file(file_id)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    local_name = safe_name(f"{ts}_{file_id}.wav")
    local_path = chat_dir / local_name

    source_ext = resolve_extension(telegram_file.file_path, ".oga" if msg.voice else ".bin")
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / f"input{source_ext}"
        await telegram_file.download_to_drive(str(input_path))
        try:
            convert_to_wav(input_path, local_path)
        except Exception:
            logger.exception("Failed converting %s to wav", input_path)
            await msg.reply_text(
                "Не удалось сконвертировать файл в WAV. Проверь, что установлен ffmpeg."
            )
            return

    probed = extract_audio_info(local_path)

    tg_duration = getattr(media, "duration", None)
    duration = probed["duration_sec"] if probed["duration_sec"] is not None else tg_duration
    sample_rate = probed["sample_rate_hz"]
    channels = probed["channels"]
    bitrate = probed["bitrate_bps"]
    mime_type = getattr(media, "mime_type", None)
    file_size = local_path.stat().st_size if local_path.exists() else getattr(media, "file_size", None)

    lines = [
        "Аудио получено, сконвертировано и сохранено в WAV.",
        f"Файл: {local_name}",
        f"Путь: {local_path}",
        f"Длительность: {format_duration(duration)}",
        f"Sample rate: {sample_rate} Hz" if sample_rate else "Sample rate: не удалось определить",
        f"Каналы: {channels}" if channels else "Каналы: не удалось определить",
        f"Битрейт: {round(bitrate / 1000, 2)} kbps" if bitrate else "Битрейт: не удалось определить",
        f"MIME: {mime_type or 'неизвестно'}",
        f"Кодек/тип: {probed['codec'] or 'неизвестно'}",
        f"Размер: {file_size} байт" if file_size else "Размер: неизвестно",
    ]

    await msg.reply_text("\n".join(lines))


async def fallback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text("Отправь voice или audio, и я верну метаданные по файлу.")


def main() -> None:
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, audio_handler))
    application.add_handler(MessageHandler(filters.ALL & ~(filters.VOICE | filters.AUDIO), fallback_handler))

    logger.info("Bot started. Polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
