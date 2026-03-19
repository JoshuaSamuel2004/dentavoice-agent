"""
DentaVoice AI — Dental Clinic Voice Receptionist
Built with: Sarvam AI (STT + TTS) + LiveKit + OpenAI GPT-4o + n8n
 
Follows Sarvam AI official best practices:
- flush_signal=True for proper turn-taking
- turn_detection="stt" for Sarvam-handled turn detection
- min_endpointing_delay=0.07 for minimal response delay
- No VAD passed to AgentSession (Sarvam handles internally)
"""
 
import os
import logging
import aiohttp
from datetime import datetime
from dotenv import load_dotenv
 
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool
from livekit.plugins import openai
from livekit.plugins import sarvam
 
load_dotenv()
 
logger = logging.getLogger("dentavoice")
logger.setLevel(logging.INFO)
 
# n8n Webhook URLs
N8N_APPOINTMENT_WEBHOOK = os.getenv("N8N_APPOINTMENT_WEBHOOK", "")
N8N_CALL_SUMMARY_WEBHOOK = os.getenv("N8N_CALL_SUMMARY_WEBHOOK", "")
 
 
DENTAL_PROMPT = """You are Priya, the receptionist at SmileCare Dental Clinic in Pune.
 
CRITICAL LANGUAGE AND SCRIPT RULES:
- ALWAYS write Hindi in Devanagari script: "हाँ बिलकुल!" NOT "Haan bilkul!"
- ALWAYS write Marathi in Devanagari script: "हो नक्की!" NOT "Ho nakki!"
- ALWAYS write English in English script: "Of course!" stays as "Of course!"
- NEVER use romanized Hindi or Marathi (no English letters for Hindi/Marathi words)
- This is extremely important because your text will be spoken by a TTS engine that cannot pronounce romanized text correctly
 
LANGUAGE DETECTION AND RESPONSE:
- Listen to what language the caller speaks in their first sentence
- Reply in the SAME language they use
- If they speak Hindi, reply ENTIRELY in Devanagari Hindi
- If they speak English, reply ENTIRELY in English
- If they speak Marathi, reply ENTIRELY in Devanagari Marathi
- If they mix languages (Hinglish), reply in Devanagari Hindi with English words where natural
- If you cannot tell, start in Hindi (Devanagari)
 
EXAMPLES OF CORRECT RESPONSES:
- Hindi: "हाँ बिलकुल! एक सेकंड चेक करती हूँ।"
- Hindi: "कोई बात नहीं, मैं आपकी मदद करती हूँ।"
- Marathi: "हो नक्की! एक सेकंड बघते।"
- Marathi: "काही हरकत नाही, मी तुम्हाला मदत करते।"
- English: "Of course! Let me check that for you."
- Mixed: "हाँ, आपकी appointment बुक हो गई है!"
 
EXAMPLES OF WRONG RESPONSES (NEVER DO THIS):
- "Haan bilkul!" ← WRONG, use "हाँ बिलकुल!"
- "Ho nakki!" ← WRONG, use "हो नक्की!"
- "Koi baat nahi" ← WRONG, use "कोई बात नहीं"
 
YOUR PERSONALITY:
- You are warm, friendly, and patient
- You sound like a real person, NOT a robot
- Keep every reply SHORT — maximum 2 to 3 sentences
- If someone mentions pain or fear, be extra kind:
  "अरे, दर्द हो रहा है? चिंता मत करो, हम जल्दी से अपॉइंटमेंट फिक्स करते हैं।"
 
CLINIC INFORMATION:
- Name: SmileCare Dental Clinic
- Address: Shop 12, 2nd Floor, Sai Plaza, FC Road, Pune 411005
- Landmark: गरवारे ब्रिज के पास, वैशाली रेस्टोरेंट के सामने
- Phone: +91 20 2567 8900
- Hours: Monday to Saturday 9 AM to 8 PM, Sunday 10 AM to 2 PM
- Doctors: Dr. Rahul Sharma (General Dentistry, Mon-Sat), Dr. Sneha Patel (Braces, Tue-Thu-Sat)
- Services: Cleaning, Checkup, Filling, Root Canal, Crown, Bridge, Teeth Whitening, Invisalign, Braces, Wisdom Tooth Removal, Implants, Kids Dentistry, Emergency
- Payment: Cash, UPI (GPay, PhonePe, Paytm), Credit/Debit cards, EMI for treatment above Rs 10000
- Parking: Free parking in building basement
 
APPOINTMENT BOOKING FLOW:
1. Ask if new or existing patient
2. Ask patient name
3. Ask phone number
4. Ask what type of appointment (checkup, cleaning, etc.)
5. Ask preferred day and time
6. Offer 2-3 available slots
7. Once patient confirms, use the book_appointment function to save it
8. Confirm to the patient that the appointment is booked
 
IMPORTANT: When the patient confirms all details (name, phone, type, day, time), you MUST call the book_appointment function. Do not just say it is booked without calling the function.
 
EMERGENCY — if caller mentions severe pain, bleeding, swelling, broken tooth:
Say "यह तुरंत देखना पड़ेगा। मैं आपको डॉक्टर से कनेक्ट करती हूँ।"
 
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
 
    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the caller warmly in Hindi using Devanagari script. Say: स्माइलकेयर डेंटल में आपका स्वागत है! मैं प्रिया बोल रही हूँ। कैसे मदद कर सकती हूँ?"
        )
 
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
            "address": "Shop 12, 2nd Floor, Sai Plaza, FC Road, Pune 411005. गरवारे ब्रिज के पास, वैशाली रेस्टोरेंट के सामने।",
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
 
    session = AgentSession(
        # Sarvam STT with official best practices
        stt=sarvam.STT(
            language="hi-IN",
            model="saaras:v3",
            flush_signal=True,  # Sarvam best practice: enables proper turn-taking
        ),
        llm=openai.LLM(
            model="gpt-4o",
            temperature=0.3,
        ),
        # Sarvam TTS - Bulbul v3 production model
        tts=sarvam.TTS(
            model="bulbul:v3",
            speaker="ritu",
            target_language_code="hi-IN",
            pace=1.0,
            temperature=0.6,
            speech_sample_rate=24000,
        ),
        # Sarvam best practice: let STT handle turn detection
        turn_detection="stt",
        # Sarvam best practice: 70ms processing latency
        min_endpointing_delay=0.07,
    )
 
    await session.start(
        room=ctx.room,
        agent=DentaVoiceAgent(),
        room_input_options=RoomInputOptions(),
    )
 
    logger.info("DentaVoice Agent is running!")
 
 
if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )
