use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, SupportedStreamConfig};
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
        let mut config = self.config.clone().config();
        config.buffer_size = cpal::BufferSize::Fixed(441 * 5);
        let mut last_samples = [0.0, 0.0];
        let stream = self.device.build_output_stream(
            &config,
            move |data: &mut [f32], _info| {
                if self.config.channels() > 2 {
                    panic!("unsupported audio config: device has more than 2 channels");
                }
                // ~50ms audio buffer
                if cons.cons.occupied_len() <= 441 * 5 * 2 {
                    return;
                }
                for s in data.chunks_mut(2) {
                    for dest in s.iter_mut() {
                        let sample = cons.cons.try_pop().unwrap_or(0);
                        *dest = (sample as f32) / (i16::MAX as f32)
                    }
                }
                // for s in data.chunks_mut(2) {
                //     let mut i = 0;
                //     let samples = [cons.cons.try_pop(), cons.cons.try_pop()].map(|src| {
                //         let res = src
                //             .map(|src| (src as f32) / (i16::MAX as f32 + 1.0))
                //             .unwrap_or(last_samples[i]);
                //         i += 1;
                //         res
                //     });
                //     for (src, dest) in samples.iter().copied().zip(s) {
                //         *dest = src;
                //     }
                //     last_samples = samples;
                // }
            },
            |err| tracing::error!("{err}"),
            None,
        );
        stream.into_diagnostic()
    }
}
