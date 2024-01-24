
from IPython.display import HTML, Audio
from google.colab.output import eval_js
import ffmpeg
from base64 import b64decode
import numpy as np
import logging
import sys


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

logging.info('Importando módulo "utils"...')


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



