from elevenlabs import generate, play, set_api_key
from openai import OpenAI
import json
from utils import get_audio, logging
from models import get_nemo_decoder_SpeakerLabelModel, get_whisper


class VirtualAssistent:
    """

    Atributos
    ---------
    STM: short_term_memory

    LTM: long_term_memory
    """

    def __init__(self, token):
        self.speaker_model = get_nemo_decoder_SpeakerLabelModel()         # usando o nemo
        self.stt_model = get_whisper() # usando o whisper

        self.client = OpenAI(api_key=token)
        
        # set_api_key('api')  # elevenlabs
        self.audio_database = {
                               'Schindler': 'schindler.wav',
                               'Thiago': 'thiago.wav'}

        self.LTM = {"Schindler": "[Schindler prefers that my talking style is similar to michael jackson]",
                    # "Gustavo": "[Gustavo prefers that I speak in PT-BR]",
                    "Thiago": ""
                    }
        self.STM = ""


        


    def generate_text_memory_write(self, expanded):
        prompt = "Long-Term Memory: " + expanded + "Short-Term Memory: " + self.STM

        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": (
                    "In your input you will receive the following sections: Short-Term Memory,Long-Term Memory. "
                    "The Short-Term Memory includes the conversation history you are having with the user, what he says "
                    "and what you reply. The entries in this section are in chronological order, where the most recent "
                    "entry is the last one. The Long-Term Memory includes the information and guidelines associated with "
                    "the specific user. Whatever information is contained here should be adhered by you, and should heavily "
                    "influence your output. Your purpose is to look at both sections, and decide what information from the "
                    "Short-Term Memory is relevant enough to be stored on the Long-Term Memory. You can update or create new "
                    "entries in the Long-Term Memory, depending on your judgement. The information you write should be short "
                    "and to the point. You should also repeat unmodified entries in your output. Your output should be the new "
                    "Long-Term Memory, with the unmodified and modified entries, formatted as a dictionary structure in python. "
                    "Formatting example: '{\"Schindler\": \"[Schindler prefers that my talking style is similar to michael jackson], "
                    "[Today me and schindler talked about apples]\"}'"
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content

    def generate_text_answer(self, specific_LTM):
        prompt = "Long-Term Memory: " + specific_LTM + " Short-Term Memory: " + self.STM

        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": (
                    "You are a helpful assistant. In your input you will receive the following sections: "
                    "Short-Term Memory, Long-Term Memory. The Short-Term Memory includes the conversation history "
                    "you are having with the user, what he says and what you reply. The entries in this section are "
                    "in chronological order, where the most recent entry is the last one. "
                    "The Long-Term Memory includes the information and guidelines associated with the specific user. "
                    "Whatever information is contained here should be adhered by you, and should heavily influence your output. "
                    "Your output should be your answer/reply, to the most recent input made by the user."
                )},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )

        return response.choices[0].message.content

    def register(self):
        name = input('Digite seu nome: ')

        audio_id = f'{name}.wav'
        get_audio(audio_id)
        self.audio_database[name] = audio_id
        self.LTM[name] = ''

        return name

    def speaker_identify(self, file='output.wav'):
        for name, audio in self.audio_database.items():
            flag = self.speaker_model.verify_speakers(file, audio)

            if flag: return name

        logging.warning('Nome não identificado, necessário fazer um cadastro')
        return self.register()

    def speech_to_text(self, file='output.wav'):
        # vad_filter -  filtra partes do áudio sem fala (a partir de 2s)
        segments, _ = self.stt_model.transcribe(file)
        text = list(segments)[0][4].strip()

        return text

    def text_to_speech(self, text):
        audio = generate(text=text,
                            voice="Rachel",
                            model="eleven_multilingual_v2"
                            )
        play(audio, notebook=True)

    def end_chat(self, specific_memory):
        logging.info('Gerando novas memórias...')
        new_memories = self.generate_text_memory_write(specific_memory)
        logging.info(f'Novas memórias: {new_memories}')

        new_memories_dict = json.loads(new_memories)
        self.LTM.update(new_memories_dict)
        self.STM = ""

    def start_chat(self):
        while True:
            try:
                # Gravar áudio
                logging.info('Gravando áudio...')
                get_audio()

                # Identificar usuário
                logging.info('Detectando nome...')
                name = self.speaker_identify()
                logging.info(f'Nome detectado: {name}')

                # Transcrever áudio
                logging.info('Transcrevendo áudio...')
                text = self.speech_to_text()
                logging.info(f'INPUT: {text}')

                # Prompt (áudio) + memória de longo prazo (personalização baseada no usuário)
                self.STM += "[" + name + " input = " + text + "]"
                specific_memory = self.LTM[name]

                # Palavras-chave para encerrar conversa e alterar a memória de longo prazo
                if ('ENCERRAR' and 'CONVERSA') in text.upper():
                    closing_text = f'Foi ótimo falar com você, {name}'
                    self.text_to_speech(closing_text)
                    self.end_chat(specific_memory)
                    break

                # Geração do output
                logging.info('Aguardando resposta...')
                output = self.generate_text_answer(specific_memory)
                self.STM += "[Assistant answer = " + output + "] "
                logging.info(f'OUTPUT: {output}')

                # Output em áudio
                logging.info('Gerando áudio...')
                self.text_to_speech(output)

            except Exception as e:
                logging.ERROR(f'Execução encerrada devido ao erro: {e}')
                break
