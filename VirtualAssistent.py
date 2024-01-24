from elevenlabs import generate, play, set_api_key
from openai import OpenAI
import json
from utils import get_audio, logging
from IPython.display import HTML, Audio
from google.colab.output import eval_js
import ffmpeg
from base64 import b64decode
import numpy as np
import logging
import sys
from faster_whisper import WhisperModel
import nemo.collections.asr as nemo_asr

def get_whisper(model_size = "large-v3"):
    return WhisperModel(model_size, device="cuda", compute_type="float16")

def get_nemo_decoder_SpeakerLabelModel(model = "nvidia/speakerverification_en_titanet_large"):
    return nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")


####################


AUDIO_HTML = """
<script>
    var my_div = document.createElement("DIV");
    var my_btn = document.createElement("BUTTON");
    var t = document.createTextNode("Press to start recording");
    
    my_btn.appendChild(t);
    my_div.appendChild(my_btn);
    document.body.appendChild(my_div);
    
    var base64data = 0;
    var reader;
    var recorder, gumStream;
    var recordButton = my_btn;
    
    var handleSuccess = function(stream) {
      gumStream = stream;
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = function(e) {
        reader = new FileReader();
        reader.readAsDataURL(e.data);
        reader.onloadend = function() {
          base64data = reader.result;
        }
      };
      recorder.start();
      };
    
    recordButton.innerText = "Recording... press to stop";
    
    navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);
    
    function toggleRecording() {
      if (recorder && recorder.state == "recording") {
          recorder.stop();
          gumStream.getAudioTracks()[0].stop();
          recordButton.innerText = "Saving the recording... pls wait!"
      }
    }
    
    function sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    var data = new Promise(resolve=>{
    recordButton.onclick = ()=>{
    toggleRecording()
    
    sleep(2000).then(() => {
      resolve(base64data.toString())
    });
    }
    });

    </script>
"""

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logging.info('Importando módulo')


def get_audio(file='output.wav'):
    """
    Grava um áudio do microfone do usuário usando a API MediaRecorder do navegador e o converte em formato WAV.

    Usa JavaScript (através do "eval_js") para criar uma interface no colab que permite ao usuário gravar áudio.
    Quando o usuário inicia e para a gravação, o áudio é capturado e transmitido para o back-end em Python.
    O áudio é então processado usando o ffmpeg para converter de binário para o formato WAV.
    A taxa de amostragem e os dados de áudio são extraídos e retornados.

    Returns
    -------
    audio : ndarray
        Um array NumPy contendo os dados de áudio gravados em formato PCM.
    sr : int
        A taxa de amostragem do áudio gravado.

    """

    display(HTML(AUDIO_HTML))

    # Pega a string base64 do áudio gravado a partir do JavaScript.
    data = eval_js("data")

    # A string é decodificada ara obter os dados binários do áudio.
    binary = b64decode(data.split(',')[1])

    process = (ffmpeg
        .input('pipe:0')
        .output(file, format='wav')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
    )
    # Transforma o áudio "bruto" em .wav
    output, err = process.communicate(input=binary)

###########



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
