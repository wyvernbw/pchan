use cpal::{
    Device, SampleFormat, Stream, SupportedStreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use miette::{Context, IntoDiagnostic, Result, bail};
use pchan_bind::{AudioConsumer, BindAudioConsumer};
use ringbuf::traits::*;

pub use cpal::Stream as AudioStream;

pub struct AudioTask {
    device: Device,
    config: SupportedStreamConfig,
    cons:   Option<AudioConsumer>,
}

impl AudioTask {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .wrap_err("failed to create audio output device")?;
        let config = device
            .supported_output_configs()
            .into_diagnostic()
            .wrap_err("failed to create default audio output config")?
            .find_map(|config_range| {
                if config_range.sample_format() == SampleFormat::F32 {
                    Some(config_range.with_sample_rate(44100))
                } else {
                    None
                }
            })
            .wrap_err("device cannot play f32 audio samples")?;
        Ok(AudioTask {
            device,
            config,
            cons: None,
        })
    }
}

impl BindAudioConsumer for AudioTask {
    fn bind_consumer(&mut self, cons: AudioConsumer) {
        self.cons = Some(cons);
    }
}

impl AudioTask {
    pub fn start(self) -> Result<Stream> {
        let stream = self.get_stream()?;
        stream.play().into_diagnostic()?;
        Ok(stream)
    }

    pub fn get_stream(self) -> Result<Stream> {
        let Some(mut cons) = self.cons else {
            bail!("audio task not bound");
        };
        let config = self.config.clone();
        let stream = self.device.build_output_stream(
            &self.config.config(),
            move |data: &mut [f32], _| {
                for ch in data.chunks_mut(config.channels() as usize) {
                    let sample = cons.cons.try_pop().unwrap_or(0);
                    let sample = (sample as f32) / (i16::MAX as f32 + 1.0);
                    for s in ch {
                        *s = sample;
                    }
                }
            },
            |err| tracing::error!("{err}"),
            None,
        );
        stream.into_diagnostic()
    }
}
