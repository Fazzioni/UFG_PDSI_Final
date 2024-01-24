from faster_whisper import WhisperModel
import nemo.collections.asr as nemo_asr



def get_whisper(model_size = "large-v3"):
    return WhisperModel(model_size, device="cuda", compute_type="float16")

def get_nemo_decoder_SpeakerLabelModel(model = "nvidia/speakerverification_en_titanet_large"):
    return nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")



