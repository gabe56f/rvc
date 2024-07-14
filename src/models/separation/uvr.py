from typing import Dict, Tuple
from pathlib import Path
import zipfile
import json

import torch
from torch import nn
import torch.nn.functional as F
from torchaudio import functional as Fa
import librosa
import numpy as np

from ...utils import spec_utils


class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, x):
        return self.conv(x)


class SeperableConv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(SeperableConv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nin,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=nin,
                bias=False,
            ),
            nn.Conv2d(nin, nout, kernel_size=1, bias=False),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU, new: bool = False
    ):
        super(Encoder, self).__init__()
        self.new = new
        if self.new:
            self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
            self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)
        else:
            self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
            self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def __call__(self, x):
        skip = self.conv1(x)
        h = self.conv2(skip)

        if self.new:
            return h
        return h, skip


class Decoder(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        ksize=3,
        stride=1,
        pad=1,
        activ=nn.ReLU,
        dropout=False,
        new: bool = False,
    ):
        super(Decoder, self).__init__()
        self.new = new
        if self.new:
            self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        else:
            self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            skip = spec_utils.crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        if self.new:
            h = self.conv1(x)
        else:
            h = self.conv(x)

        if self.dropout is not None:
            h = self.dropout(h)

        return h


class ASPPModule(nn.Module):
    def __init__(self, nin, nout, dilations=(4, 8, 16), activ=nn.ReLU):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ),
        )
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        self.conv3 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[0], dilations[0], activ=activ
        )
        self.conv4 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[1], dilations[1], activ=activ
        )
        self.conv5 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(nin * 5, nout, 1, 1, 0, activ=activ), nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        bottle = self.bottleneck(out)
        return bottle


class ASPPModule_new(nn.Module):
    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super(ASPPModule_new, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ),
        )
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv3 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[0], dilations[0], activ=activ
        )
        self.conv4 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[1], dilations[1], activ=activ
        )
        self.conv5 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[2], dilations[2], activ=activ
        )
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.bottleneck(out)
        if self.dropout is not None:
            out = self.dropout(out)

        return out


class LSTMModule(nn.Module):
    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super(LSTMModule, self).__init__()

        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        self.lstm = nn.LSTM(
            input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True
        )
        self.dense = nn.Sequential(
            nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU()
        )

    def forward(self, x):
        N, _, nbins, nframes = x.size()
        h = self.conv(x)[:, 0]  # N, nbins, nframes
        h = h.permute(2, 0, 1)  # nframes, N, nbins
        h, _ = self.lstm(h)
        h = self.dense(h.reshape(-1, h.size()[-1]))  # nframes * N, nbins
        h = h.reshape(nframes, N, 1, nbins)
        h = h.permute(1, 2, 3, 0)

        return h


# -- old? --


class BaseASPPNet(nn.Module):
    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.enc1 = Encoder(nin, ch, 3, 2, 1)
        self.enc2 = Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = Encoder(ch * 4, ch * 8, 3, 2, 1)

        self.aspp = ASPPModule(ch * 8, ch * 16, dilations)

        self.dec4 = Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        h = self.aspp(h)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        self.stg1_low_band_net = BaseASPPNet(2, 32)
        self.stg1_high_band_net = BaseASPPNet(2, 32)

        self.stg2_bridge = Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(16, 32)

        self.stg3_bridge = Conv2DBNActiv(66, 32, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(32, 64)

        self.out = nn.Conv2d(64, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(32, 2, 1, bias=False)

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.offset = 128

    def forward(self, x, aggressiveness=None):
        mix = x.detach()
        x = x.clone()

        x = x[:, :, : self.max_bin]

        bandw = x.size()[2] // 2
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:]),
            ],
            dim=2,
        )

        h = torch.cat([x, aux1], dim=1)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        h = torch.cat([x, aux1, aux2], dim=1)
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        mask = torch.sigmoid(self.out(h))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        if self.training:
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode="replicate",
            )
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode="replicate",
            )
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            if aggressiveness:
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]],
                    1 + aggressiveness["value"] / 3,
                )
                mask[:, :, aggressiveness["split_bin"] :] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"] :],
                    1 + aggressiveness["value"],
                )

            return mask * mix

    def predict(self, x_mag, aggressiveness=None):
        h = self.forward(x_mag, aggressiveness)

        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
            assert h.size()[3] > 0

        return h


# -- new? --


class BaseNet(nn.Module):
    def __init__(
        self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))
    ):
        super(BaseNet, self).__init__()
        self.enc1 = Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = Encoder(nout, nout * 2, 3, 2, 1, new=True)
        self.enc3 = Encoder(nout * 2, nout * 4, 3, 2, 1, new=True)
        self.enc4 = Encoder(nout * 4, nout * 6, 3, 2, 1, new=True)
        self.enc5 = Encoder(nout * 6, nout * 8, 3, 2, 1, new=True)

        self.aspp = ASPPModule_new(nout * 8, nout * 8, dilations, dropout=True)

        self.dec4 = Decoder(nout * (6 + 8), nout * 6, 3, 1, 1, new=True)
        self.dec3 = Decoder(nout * (4 + 6), nout * 4, 3, 1, 1, new=True)
        self.dec2 = Decoder(nout * (2 + 4), nout * 2, 3, 1, 1, new=True)
        self.lstm_dec2 = LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1, new=True)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = self.aspp(e5)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)

        return h


class CascadedNet(nn.Module):
    def __init__(self, n_fft, nout=32, nout_lstm=128):
        super(CascadedNet, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64

        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm),
            Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0),
        )

        self.stg1_high_band_net = BaseNet(
            2, nout // 4, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm),
            Conv2DBNActiv(nout, nout // 2, 1, 1, 0),
        )
        self.stg2_high_band_net = BaseNet(
            nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg3_full_band_net = BaseNet(
            3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm
        )

        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

    def forward(self, x):
        x = x[:, :, : self.max_bin]

        bandw = x.size()[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)

        mask = torch.sigmoid(self.out(f3))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        if self.training:
            aux = torch.cat([aux1, aux2], dim=1)
            aux = torch.sigmoid(self.aux_out(aux))
            aux = F.pad(
                input=aux,
                pad=(0, 0, 0, self.output_bin - aux.size()[2]),
                mode="replicate",
            )
            return mask, aux
        else:
            return mask

    def predict_mask(self, x):
        mask = self.forward(x)

        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]
            assert mask.size()[3] > 0

        return mask

    def predict(self, x, aggressiveness=None):
        mask = self.forward(x)
        pred_mag = x * mask

        if self.offset > 0:
            pred_mag = pred_mag[:, :, :, self.offset : -self.offset]
            assert pred_mag.size()[3] > 0

        return pred_mag


def int_key_selector(d: Dict[str, str]):
    r = {}
    for k, v in d:
        if k.isdigit():
            k = int(k)
        r[k] = v
    return r


class UVRParameters(object):
    def __init__(self, config_path: str, device, dtype):
        config = Path(config_path)
        self.device = device
        self.dtype = dtype
        if config.suffix == ".pth":
            with zipfile.ZipFile(config, "r") as z:
                self.param = self.parse_model_data(z.read("config.json"))
        elif config.suffix == ".json":
            with open(config, "r") as f:
                self.param = self.parse_model_data(f.read())
        else:
            self.param = self.construct_default_model_data()

    def construct_default_model_data(self):
        return {
            "bins": 768,
            "sr": 44100,
            "pre_filter_start": 757,
            "pre_filter_stop": 768,
            "band": {
                1: {
                    "sr": 11025,
                    "hl": 128,
                    "n_fft": 960,
                    "crop_start": 0,
                    "crop_stop": 245,
                    "lpf_start": 61,
                    "res_type": "polyphase",
                },
                2: {
                    "sr": 44100,
                    "hl": 512,
                    "n_fft": 1536,
                    "crop_start": 24,
                    "crop_stop": 547,
                    "hpf_start": 81,
                    "res_type": "sinc_best",
                },
            },
        }

    def parse_model_data(self, data: str):
        r: dict = json.loads(data, object_pairs_hook=int_key_selector)
        for k in [
            "mid_side",
            "mid_side_b",
            "mid_side_b2",
            "stereo_w",
            "stereo_n",
            "reverse",
        ]:
            if k not in r.keys():
                r[k] = False
        return r


class UVR:
    def __init__(self, agg, model_path, device, dtype, new: bool = False):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.data = {
            "postprocess": False,
            "tta": False,
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }

        if new:
            params = UVRParameters("uvr/4band_v3.json", device, dtype)
            model = CascadedNet(
                params.param["bins"] * 2, 64 if "DeReverb" in model_path else 48
            )
        else:
            params = UVRParameters("uvr/4band_v2.json", device, dtype)
            model = CascadedASPPNet(params.param["bins"] * 2)
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt)
        model.to(device, dtype)
        model.eval()

        self.params = params
        self.model = model

    def to(self, device, dtype) -> "UVR":
        self.device = device
        self.dtype = dtype
        self.params.device = device
        self.params.dtype = dtype
        self.model.to(device, dtype)
        return self

    def __call__(self, input: np.ndarray) -> Tuple[int, Dict[str, np.ndarray]]:
        torch.cuda.empty_cache()

        # safe to assume input is "high quality" -- 44.1khz input.
        X_wave, X_spec_s = {}, {}
        num_bands = len(self.params.param["band"])
        for band in range(num_bands, 0, -1):
            band_params = self.params.param["band"][band]
            if num_bands == band:
                # input IS the first band
                if isinstance(input, str):
                    X_wave[band], _ = librosa.load(input, sr=band_params["sr"])
                else:
                    X_wave[band] = input
                if X_wave[band].ndim == 1:
                    X_wave[band] = np.asfortranarray([X_wave[band], X_wave[band]])
            else:
                X_wave[band] = (
                    Fa.resample(
                        torch.from_numpy(X_wave[band + 1]),
                        self.params.param["band"][band + 1]["sr"],
                        band_params["sr"],
                    )
                    .cpu()
                    .float()
                    .numpy()
                )
            X_spec_s[band] = spec_utils.wave_to_spectrogram(
                X_wave[band],
                band_params["hl"],
                band_params["n_fft"],
                self.params.param["mid_side"],
                self.params.param["mid_side_b2"],
                self.params.param["reverse"],
            )
            if num_bands == band and self.data["high_end_process"] != "none":
                input_high_end_h = (
                    band_params["n_fft"] // 2 - band_params["crop_stop"]
                ) + (
                    self.params.param["pre_filter_stop"]
                    - self.params.param["pre_filter_start"]
                )
                input_high_end = X_spec_s[band][
                    :,
                    band_params["n_fft"] // 2
                    - input_high_end_h : band_params["n_fft"] // 2,
                    :,
                ]
        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.params)
        aggressive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggressive_set,
            "split_bin": self.params.param["band"][1]["crop_stop"],
        }
        with torch.no_grad():
            pred, X_mag, X_phase = _inference(
                X_spec_m, self.device, self.dtype, self.model, aggressiveness, self.data
            )
        if self.data["postprocess"]:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring(
                self.data["high_end_process"], y_spec_m, input_high_end, self.params
            )
            accompaniment = spec_utils.cmb_spectrogram_to_wave(
                y_spec_m, self.params, input_high_end_h, input_high_end_
            )
        else:
            accompaniment = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.params)
        accompaniment = np.array(accompaniment)

        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring(
                self.data["high_end_process"], v_spec_m, input_high_end, self.params
            )
            vocals = spec_utils.cmb_spectrogram_to_wave(
                v_spec_m, self.params, input_high_end_h, input_high_end_
            )
        else:
            vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.params)
        vocals = np.array(vocals)

        return self.params.param["band"][num_bands]["sr"], {
            "vocals": vocals,
            "accompaniment": accompaniment,
        }


def _inference(X_spec, device, dtype, model, aggressiveness, data):
    def make_padding(width, cropsize, offset):
        left = offset
        roi_size = cropsize - left * 2
        if roi_size == 0:
            roi_size = cropsize
        right = roi_size - (width % roi_size) + left

        return left, right, roi_size

    def _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness, dtype):
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(n_window):
                start = i * roi_size
                X_mag_window = X_mag_pad[
                    None, :, :, start : start + data["window_size"]
                ]
                X_mag_window = torch.from_numpy(X_mag_window)
                X_mag_window = X_mag_window.to(device, dtype)

                pred = model.predict(X_mag_window, aggressiveness)

                pred = pred.detach().cpu().float().numpy()
                preds.append(pred[0])

            pred = np.concatenate(preds, axis=2)
        return pred

    def preprocess(X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    X_mag, X_phase = preprocess(X_spec)

    coef = X_mag.max()
    X_mag_pre = X_mag / coef

    n_frame = X_mag_pre.shape[2]
    pad_l, pad_r, roi_size = make_padding(n_frame, data["window_size"], model.offset)
    n_window = int(np.ceil(n_frame / roi_size))

    X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

    pred = _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness, dtype)
    pred = pred[:, :, :n_frame]

    if data["tta"]:
        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

        pred_tta = _execute(
            X_mag_pad, roi_size, n_window, device, model, aggressiveness, dtype
        )
        pred_tta = pred_tta[:, :, roi_size // 2 :]
        pred_tta = pred_tta[:, :, :n_frame]

        return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.0j * X_phase)
    else:
        return pred * coef, X_mag, np.exp(1.0j * X_phase)
