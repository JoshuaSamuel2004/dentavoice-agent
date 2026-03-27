"""
DentaVoice AI — Dental Clinic Voice Receptionist
Built with: Sarvam AI (STT + TTS) + LiveKit + OpenAI GPT-4o + n8n

FIXES APPLIED:
1. MIC SENSITIVITY / NOISE INTERRUPTIONS:
   - Added Silero VAD with higher threshold (0.65) to filter ambient noise
   - Increased min_endpointing_delay to 0.5s so brief sounds don't trigger turns
   - Added max_endpointing_delay of 5.0s for longer pauses
   - Set min_interruption_duration to 0.8s — short noises won't interrupt agent speech
   - Set min_interruption_words to 1 — must detect at least 1 real word to interrupt
   - Enabled noise cancellation via RoomInputOptions

2. ENGLISH TIMES SPOKEN IN HINDI:
   - TTS target_language_code was hardcoded to "hi-IN"
   - Now dynamically switches TTS language using stt_node override + update_options()
   - When user speaks English → TTS switches to "en-IN"
   - When user speaks Hindi → TTS switches to "hi-IN"
   - When user speaks Marathi → TTS switches to "mr-IN"

3. "HELLO" RESTARTING CONVERSATION:
   - The on_enter greeting was unconditional and could re-trigger
   - Added prompt instructions telling the LLM: if user says "hello" mid-conversation,
     just acknowledge and continue — do NOT restart the greeting flow
   - The generate_reply in on_enter only fires once when agent first enters
"""

import os
import logging
import aiohttp
from typing import AsyncIterable
from datetime import datetime
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, ModelSettings, stt
from livekit.plugins import openai
from livekit.plugins import sarvam
from livekit.plugins import silero
from livekit.plugins import noise_cancellation

load_dotenv()

logger = logging.getLogger("dentavoice")
logger.setLevel(logging.INFO)

# n8n Webhook URLs
N8N_APPOINTMENT_WEBHOOK = os.getenv("N8N_APPOINTMENT_WEBHOOK", "")
N8N_CALL_SUMMARY_WEBHOOK = os.getenv("N8N_CALL_SUMMARY_WEBHOOK", "")

# Language code mapping: STT detected language → Sarvam TTS language code
STT_TO_SARVAM_TTS = {
    "hi": "hi-IN",
    "en": "en-IN",
    "mr": "mr-IN",
    "hi-IN": "hi-IN",
    "en-IN": "en-IN",
    "mr-IN": "mr-IN",
}

DEFAULT_TTS_LANGUAGE = "hi-IN"


DENTAL_PROMPT = """You are Priya, the receptionist at SmileCare Dental Clinic in Pune.

CRITICAL LANGUAGE AND SCRIPT RULES:
- ALWAYS write Hindi in Devanagari script: "हाँ बिलकुल!" NOT "Haan bilkul!"
- ALWAYS write Marathi in Devanagari script: "हो नक्की!" NOT "Ho nakki!"
- ALWAYS write English in English script: "Of course!" stays as "Of course!"
- NEVER use romanized Hindi or Marathi (no English letters for Hindi/Marathi words)
- This is extremely important because your text will be spoken by a TTS engine that cannot pronounce romanized text correctly

LANGUAGE DETECTION AND RESPONSE:
- Listen carefully to what language the caller speaks
- Reply in the EXACT SAME language they use
- If they speak Hindi, reply ENTIRELY in Devanagari Hindi
- If they speak English, reply ENTIRELY in English
- If they speak Marathi, reply ENTIRELY in Devanagari Marathi
- If they mix languages (Hinglish), reply in Devanagari Hindi with English words where natural
- If you cannot tell the language, start in Hindi (Devanagari)

CRITICAL LANGUAGE SWITCHING RULES:
- IMPORTANT: Once a patient switches language mid-conversation, you MUST switch too and continue in their new language
- If patient speaks English, reply ONLY in English — do NOT mix Hindi words
- If patient speaks Marathi, reply ONLY in Devanagari Marathi — do NOT mix Hindi words
- If patient speaks Hindi, reply ONLY in Devanagari Hindi
- When saying times, dates, and numbers, say them in the SAME language as the rest of your reply
- In English: say "Monday at 11 AM" NOT "सोमवार 11 बजे"
- In Hindi: say "सोमवार सुबह 11 बजे" NOT "Monday 11 AM"
- In Marathi: say "सोमवारी सकाळी 11 वाजता" NOT "Monday 11 AM"
- Doctor names stay the same in all languages: "Dr. Rahul Sharma", "Dr. Sneha Patel"

CRITICAL TIME AND NUMBER FORMATTING RULES:
- When replying in English, ALL times must be in English: "9 AM", "11 AM", "3 PM", "8 PM"
- NEVER write "9 बजे" or "11 बजे" when replying in English
- NEVER write AM/PM in Devanagari when replying in English
- When replying in Hindi, ALL times must be in Hindi: "सुबह 9 बजे", "दोपहर 11 बजे", "शाम 3 बजे"
- When replying in Marathi, ALL times must be in Marathi: "सकाळी 9 वाजता", "दुपारी 11 वाजता"
- This is critical because the TTS engine reads the text exactly as written

EXAMPLES OF CORRECT LANGUAGE SWITCHING:
Patient says in English: "I want to book an appointment"
You reply in English: "Of course! Are you a new patient or an existing one? And what type of appointment do you need?"

Patient says in Hindi: "मुझे अपॉइंटमेंट बुक करनी है"
You reply in Hindi: "हाँ बिलकुल! क्या आप नए मरीज़ हैं या पहले से आ चुके हैं? और किस प्रकार की अपॉइंटमेंट चाहिए?"

Patient says in Marathi: "मला एक अपॉइंटमेंट बुक करायची आहे"
You reply in Marathi: "हो नक्की! तुम्ही नवीन पेशंट आहात का? आणि कोणत्या प्रकारची अपॉइंटमेंट हवी आहे?"

EXAMPLES OF WRONG RESPONSES (NEVER DO THIS):
- Patient speaks English, you reply in Hindi ← WRONG
- Patient speaks Marathi, you reply in Hindi ← WRONG
- Patient speaks English, you say times in Hindi like "सोमवार 11 बजे" ← WRONG
- "Haan bilkul!" ← WRONG, use "हाँ बिलकुल!"
- "Ho nakki!" ← WRONG, use "हो नक्की!"
- In English reply: "Monday to Saturday 9 बजे to 8 बजे" ← WRONG, say "Monday to Saturday 9 AM to 8 PM"

YOUR PERSONALITY:
- You are warm, friendly, and patient
- You sound like a real person, NOT a robot
- Keep every reply SHORT — maximum 1 to 2 sentences
- NEVER ask more than ONE question at a time. Wait for the answer before asking the next question.
- This is a voice call, not a text chat. The patient can only answer one thing at a time.
- If someone mentions pain or fear, be extra kind

CRITICAL CONVERSATION CONTINUITY RULE:
- If a patient says "hello", "hi", "hey", or any greeting DURING an ongoing conversation, DO NOT restart the conversation or re-introduce yourself.
- Simply acknowledge warmly and continue from where the conversation left off.
- Example: If you were in the middle of booking an appointment and the patient says "hello", respond with something like "Yes, I'm here! So shall we continue with the appointment?" — do NOT say the welcome greeting again.
- The welcome greeting ("SmileCare Dental mein aapka swagat hai...") should ONLY happen at the very start of the call, never again.

CLINIC INFORMATION:
- Name: SmileCare Dental Clinic
- Address: Shop 12, 2nd Floor, Sai Plaza, FC Road, Pune 411005
- Landmark: Near Garware Bridge, opposite Vaishali Restaurant
- Phone: +91 20 2567 8900
- Hours: Monday to Saturday 9 AM to 8 PM, Sunday 10 AM to 2 PM
- Doctors: Dr. Rahul Sharma (General Dentistry, Mon-Sat), Dr. Sneha Patel (Braces, Tue-Thu-Sat)
- Services: Cleaning, Checkup, Filling, Root Canal, Crown, Bridge, Teeth Whitening, Invisalign, Braces, Wisdom Tooth Removal, Implants, Kids Dentistry, Emergency
- Payment: Cash, UPI (GPay, PhonePe, Paytm), Credit/Debit cards, EMI for treatment above Rs 10000
- Parking: Free parking in building basement

APPOINTMENT BOOKING FLOW:
Ask these ONE AT A TIME. Never combine two questions. Wait for each answer before moving to the next.
1. First ask: new or existing patient? (wait for answer)
2. Then ask: what is your name? (wait for answer)
3. Then ask: what is your phone number? (wait for answer)
   - If patient says "this is the same number" or "use this number" or does not say actual digits, ask again politely: "Could you please         tell me the 10-digit number? I need to save it for your appointment confirmation."
   - Do NOT proceed without getting actual digits. The phone number must be a real 10-digit Indian number.
   - Do NOT assume or make up a phone number.
4. Then ask: what type of appointment? (wait for answer)
5. Then ask: which day do you prefer? (wait for answer)
6. Then offer 2-3 time slots (wait for answer)
7. Once patient confirms, call the book_appointment function
8. Confirm the booking
NEVER combine questions like "What is your name and phone number?" — ask them separately.

IMPORTANT: When the patient confirms all details (name, phone, type, day, time), you MUST call the book_appointment function. Do not just say it is booked without calling the function.

EMERGENCY RESPONSES:
- In Hindi: "यह तुरंत देखना पड़ेगा। मैं आपको डॉक्टर से कनेक्ट करती हूँ।"
- In English: "This needs immediate attention. Let me connect you with the doctor right away."
- In Marathi: "हे लगेच बघावे लागेल. मी तुम्हाला डॉक्टरांशी जोडते."

RULES:
- NEVER give medical advice
- NEVER share other patients info
- NEVER promise exact prices (give ranges only)
- If you dont know something, say you will call back
- NEVER make up information
- Always ask for patient name and phone number before booking
"""


class DentaVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=DENTAL_PROMPT,
        )
        self._current_tts_language = DEFAULT_TTS_LANGUAGE

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the caller warmly in Hindi using Devanagari script. Say: स्माइलकेयर डेंटल में आपका स्वागत है! मैं प्रिया बोल रही हूँ। कैसे मदद कर सकती हूँ?"
        )

    # ─── FIX #2: Dynamic TTS language switching ───────────────────────
    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterable[stt.SpeechEvent]:
        """
        Override the STT node to intercept speech events and detect language.
        When the detected language changes, dynamically update the Sarvam TTS
        target_language_code so English text is spoken in English, not Hindi.
        """
        async for event in Agent.default.stt_node(self, audio, model_settings):
            # Check for transcript events that carry language info
            if (
                event.type in (
                    stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    stt.SpeechEventType.FINAL_TRANSCRIPT,
                )
                and event.alternatives
            ):
                detected_lang = event.alternatives[0].language
                if detected_lang:
                    self._update_tts_language(detected_lang)

            yield event

    def _update_tts_language(self, detected_language: str) -> None:
        """
        Map STT-detected language code to Sarvam TTS language code and
        call update_options() if the language has changed.
        """
        # Sarvam STT may return "hi", "en", "mr", "hi-IN", "en-IN", etc.
        base_lang = detected_language.split("-")[0].lower()
        tts_lang = STT_TO_SARVAM_TTS.get(detected_language, STT_TO_SARVAM_TTS.get(base_lang))

        if tts_lang and tts_lang != self._current_tts_language:
            logger.info(
                f"Language switch detected: {self._current_tts_language} → {tts_lang} "
                f"(STT reported: {detected_language})"
            )
            self._current_tts_language = tts_lang
            self.session.tts.update_options(target_language_code=tts_lang)

    # ─── Tools ────────────────────────────────────────────────────────

    @function_tool()
    async def book_appointment(
        self,
        patient_name: str,
        phone_number: str,
        appointment_type: str,
        preferred_day: str,
        preferred_time: str,
        is_new_patient: bool = True,
        notes: str = "",
    ) -> str:
        """Book a dental appointment for a patient. Call this when the patient has confirmed their name, phone number, appointment type, preferred day, and time.

        Args:
            patient_name: Full name of the patient
            phone_number: Patient phone number with country code
            appointment_type: Type of dental appointment like Checkup, Cleaning, Root Canal, etc.
            preferred_day: Day of the week for the appointment like Monday, Tuesday, etc.
            preferred_time: Time for the appointment like 11 AM, 3 PM, etc.
            is_new_patient: Whether this is a new patient or existing patient
            notes: Any additional notes about the appointment
        """
        logger.info(
            f"Booking appointment: {patient_name}, {phone_number}, "
            f"{appointment_type}, {preferred_day} {preferred_time}"
        )

        appointment_data = {
            "patient_name": patient_name,
            "phone_number": phone_number,
            "appointment_type": appointment_type,
            "preferred_day": preferred_day,
            "preferred_time": preferred_time,
            "is_new_patient": is_new_patient,
            "notes": notes,
            "booked_at": datetime.now().isoformat(),
            "status": "confirmed",
            "clinic": "SmileCare Dental Clinic",
        }

        # Send to n8n webhook
        if N8N_APPOINTMENT_WEBHOOK:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        N8N_APPOINTMENT_WEBHOOK,
                        json=appointment_data,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Appointment logged to Google Sheets: {patient_name}")
                        else:
                            logger.error(f"n8n webhook error: {response.status}")
            except Exception as e:
                logger.error(f"Failed to send to n8n: {e}")
        else:
            logger.warning("N8N_APPOINTMENT_WEBHOOK not set, skipping webhook")

        return (
            f"Appointment booked successfully for {patient_name} on "
            f"{preferred_day} at {preferred_time} for {appointment_type}."
        )

    @function_tool()
    async def get_clinic_info(
        self,
        info_type: str,
    ) -> str:
        """Look up clinic information such as address, hours, services, doctors, payment options, or parking details.

        Args:
            info_type: Type of information requested like address, hours, doctors, services, payment, or parking
        """
        info = {
            "address": "Shop 12, 2nd Floor, Sai Plaza, FC Road, Pune 411005. Near Garware Bridge, opposite Vaishali Restaurant.",
            "hours": "Monday to Saturday: 9 AM to 8 PM. Sunday: 10 AM to 2 PM.",
            "doctors": "Dr. Rahul Sharma (General Dentistry, Mon-Sat), Dr. Sneha Patel (Braces, Tue-Thu-Sat).",
            "services": "Cleaning, Checkup, Filling, Root Canal, Crown, Bridge, Teeth Whitening, Invisalign, Braces, Wisdom Tooth Removal, Implants, Kids Dentistry, Emergency.",
            "payment": "Cash, UPI (GPay, PhonePe, Paytm), Credit/Debit cards. EMI available for treatments above Rs 10,000.",
            "parking": "Free parking available in the building basement.",
        }

        info_type_lower = info_type.lower()
        for key, value in info.items():
            if key in info_type_lower:
                return value

        return "I have information about: address, hours, doctors, services, payment options, and parking."


async def entrypoint(ctx: agents.JobContext):
    logger.info("DentaVoice Agent starting...")

    await ctx.connect()

    # ─── FIX #1: Preload Silero VAD for noise filtering ──────────────
    vad_instance = silero.VAD.load(
        min_speech_duration=0.15,    # Ignore sounds shorter than 150ms
        min_silence_duration=0.4,    # Need 400ms silence to consider speech ended
        activation_threshold=0.65,   # Higher threshold = less sensitive to noise
                                     # Default is 0.5; 0.65 filters out more ambient noise
    )

    session = AgentSession(
        # Sarvam STT with auto language detection
        stt=sarvam.STT(
            language="unknown",
            model="saaras:v3",
            flush_signal=True,
        ),
        llm=openai.LLM(
            model="gpt-4o",
            temperature=0.3,
        ),
        # Sarvam TTS — starts with Hindi, switches dynamically via stt_node
        tts=sarvam.TTS(
            model="bulbul:v3",
            speaker="ritu",
            target_language_code="hi-IN",
            pace=1.0,
            temperature=0.6,
            speech_sample_rate=24000,
        ),
        # ─── Silero VAD for noise filtering ───────────────────────────
        vad=vad_instance,
        # Use VAD-based turn detection instead of raw STT endpointing
        # This means VAD must confirm speech ended before a turn is committed,
        # which filters out random noises that STT might pick up
        turn_detection="vad",
        # ─── FIX #1 continued: Tuned endpointing & interruption params ─
        min_endpointing_delay=0.5,          # Wait 500ms of silence before ending turn
                                             # (was 0.07 — way too aggressive)
        max_endpointing_delay=5.0,           # Allow up to 5s for thinking pauses
        min_interruption_duration=0.8,       # Must speak 800ms to interrupt agent
                                             # (prevents coughs/clicks from interrupting)
        min_interruption_words=1,            # Must detect at least 1 word to count
                                             # as a real interruption
        # ─── FIX #3: Prevent false re-triggers on "hello" ────────────
        # false_interruption_timeout helps: if user says a short word like "hello"
        # while agent is speaking and then goes silent, agent resumes its speech
        # instead of restarting
        false_interruption_timeout=2.0,
        resume_false_interruption=True,
    )

    await session.start(
        room=ctx.room,
        agent=DentaVoiceAgent(),
        # ─── FIX #1 continued: Enable noise cancellation on input audio ─
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    logger.info("DentaVoice Agent is running!")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )
