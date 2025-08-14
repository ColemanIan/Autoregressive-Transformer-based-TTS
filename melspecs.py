import torch
import torchaudio
import torchaudio.transforms as T


class Melspecs(torch.nn.Module):
    
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu' #store device for pytorch module, needs a parameter to save
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=hp.sample_rate,
            n_fft=hp.n_fft,
            win_length=hp.win_length,
            hop_length=hp.hop_length,
            n_mels=hp.n_mels,
            power=2.0
        )
        
        self.amp_to_db = T.AmplitudeToDB(top_db=hp.top_db)
        
        self.inverse_mel = T.InverseMelScale(
            n_stft=hp.n_stft,
            n_mels=hp.n_mels,
            sample_rate=hp.sample_rate
        )
        
        self.griffin_lim = T.GriffinLim(
            n_fft=hp.n_fft,
            win_length=hp.win_length,
            hop_length=hp.hop_length,
            power=2.0
        )
    
    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        # Move all transforms to device
        self.mel_spec = self.mel_spec.to(device)
        self.amp_to_db = self.amp_to_db.to(device)
        self.inverse_mel = self.inverse_mel.to(device)
        self.griffin_lim = self.griffin_lim.to(device)
        return super().to(device)

    def wav_to_mel(self, wav_path):
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.to(self._device)
        
        if sr != self.hp.sample_rate:
            resample = T.Resample(sr, self.hp.sample_rate).to(self._device)
            waveform = resample(waveform)
            
        mel = self.mel_spec(waveform)
        mel_db = self.amp_to_db(mel)
        return mel_db

    def mel_to_wav(self, mel_db):
        mel_amp = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=1.0)
        linear = self.inverse_mel(mel_amp)
        waveform = self.griffin_lim(linear)
        return waveform