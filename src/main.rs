#![feature(int_log)]
use crossbeam_channel::unbounded;
use image::{ImageBuffer, ImageFormat, RgbImage};
use num_complex::Complex64;
use std::num::ParseFloatError;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread;
use std::time::Duration;
use structopt::StructOpt;

const BLACK_RGB: image::Rgb<u8> = image::Rgb([40, 42, 54]);

/// Mandelbrot set image generator.
#[derive(StructOpt, Debug)]
#[structopt(name = "mandelbrust")]
struct Opt {
    /// Path of the output file
    #[structopt(name = "FILE", parse(from_os_str))]
    file: PathBuf,

    /// Iteration limit, used to determine wether a point converges.
    #[structopt(short = "l", long, default_value = "500")]
    iter_limit: u64,

    /// width of the generated image.
    #[structopt(long, default_value = "500")]
    width: u32,

    /// height of the generated image.
    #[structopt(long, default_value = "500")]
    height: u32,

    /// Number of threads to use.
    #[structopt(long)]
    threads: Option<usize>,

    /// The center of the image
    #[structopt(long, default_value = "0,0", parse(try_from_str=parse_complex))]
    center: Complex64,

    /// The span of the image
    #[structopt(long, default_value = "2")]
    span: f64,

    /// Threshold where task splitting should not be triggered, should be above 5
    #[structopt(long, default_value = "21")]
    thresh: u32,

    #[structopt(long)]
    bars: bool,
}

enum ImgChange {
    Changes(Vec<(u32, u32, Option<u64>)>),
    Fill {
        start_x: u32,
        start_y: u32,
        stop_x: u32,
        stop_y: u32,
    },
}

fn parse_complex(s: &str) -> Result<Complex64, ParseFloatError> {
    if let Some(div) = s.find(',') {
        let re = f64::from_str(&s[..div])?;
        let im = f64::from_str(&s[div + 1..])?;
        Ok(Complex64 { re, im })
    } else {
        Ok(Complex64::new(2., 2.))
    }
}

#[derive(Debug)]
struct Task {
    start_x: u32,
    start_y: u32,
    stop_x: u32,
    stop_y: u32,
    top_left: Complex64,
    bot_right: Complex64,
    glob_width: u32,
    glob_height: u32,
    iter_limit: u64,
    divide_threshold: u32,
}

fn main() {
    let opt = Opt::from_args();
    let mut img: RgbImage = ImageBuffer::new(opt.width, opt.height);
    let (work_tx, work_rx) = unbounded();
    let (img_tx, img_rx) = channel();

    let span = Complex64::new(opt.span, opt.span);
    let top_left = opt.center + span;
    let bot_right = opt.center - span;
    let active_count = Arc::new(RwLock::new(0));

    work_tx
        .send(Task {
            start_x: 0,
            start_y: 0,
            stop_x: opt.width - 1,
            stop_y: opt.height - 1,
            top_left,
            bot_right,
            glob_width: opt.width,
            glob_height: opt.height,
            iter_limit: opt.iter_limit,
            divide_threshold: if opt.thresh >= 5 { opt.thresh } else { 5 },
        })
        .unwrap();
    *(active_count.write().unwrap()) += 1;

    let mut threads: Vec<_> = vec![];

    for _ in 0..opt.threads.unwrap_or(num_cpus::get()) {
        let w_tx = work_tx.clone();
        let w_rx = work_rx.clone();
        let out = img_tx.clone();
        let active_count = Arc::clone(&active_count);
        threads.push(thread::spawn(move || loop {
            let job;
            {
                let count = active_count.read().unwrap();
                if *count == 0 {
                    break;
                } else {
                    job = w_rx.try_recv();
                }
            }

            if let Ok(task) = job {
                let start_x = task.start_x;
                let start_y = task.start_y;
                let stop_x = task.stop_x;
                let stop_y = task.stop_y;

                let (modification, subtasks) = process_task(task);

                out.send(modification).unwrap();

                if let Some(subtasks) = subtasks {
                    *(active_count.write().unwrap()) += 4;
                    for task in subtasks {
                        w_tx.send(task).unwrap();
                    }
                } else {
                    out.send(ImgChange::Fill {
                        start_x: start_x + 1,
                        start_y: start_y + 1,
                        stop_x: stop_x - 1,
                        stop_y: stop_y - 1,
                    })
                    .unwrap();
                }
                *(active_count.write().unwrap()) -= 1;
            } else {
                thread::sleep(Duration::new(0, 10_000_000))
            }
        }));
    }

    loop {
        match img_rx.try_recv() {
            Ok(ImgChange::Changes(changes)) => {
                for (x, y, color) in changes {
                    if opt.bars {
                        img.put_pixel(x, y, image::Rgb([255, 255, 255]));
                    } else {
                        if let Some(div) = color {
                            img.put_pixel(
                                x,
                                y,
                                image::Rgb(gradient(
                                    &[68, 71, 90],
                                    &[139, 233, 253],
                                    div,
                                    opt.iter_limit / 10,
                                )),
                            );
                        } else {
                            img.put_pixel(x, y, BLACK_RGB);
                        }
                    }
                }
            }
            Ok(ImgChange::Fill {
                start_x,
                start_y,
                stop_x,
                stop_y,
            }) => {
                for i in start_x..stop_x + 1 {
                    for j in start_y..stop_y + 1 {
                        img.put_pixel(i, j, BLACK_RGB);
                    }
                }
            }
            _ => {
                if *(active_count.read().unwrap()) == 0 {
                    break;
                } else {
                    thread::sleep(Duration::new(0, 10_000_000))
                }
            }
        }
    }

    for t in threads {
        t.join().unwrap();
    }

    drop(work_rx);
    drop(work_tx);
    drop(img_rx);
    drop(img_tx);

    img.save_with_format(opt.file, ImageFormat::Bmp).unwrap();
}

fn process_task(t: Task) -> (ImgChange, Option<[Task; 4]>) {
    // Compute surrouding rectangle
    let mut should_split = false;
    let mut all_modifs = Vec::new();
    let delta_x = t.stop_x - t.start_x;
    let delta_y = t.stop_y - t.start_y;

    if delta_x < t.divide_threshold || delta_y < t.divide_threshold {
        let (changes, _) = compute_for_range(&t, t.start_x, t.stop_x, t.start_y, t.stop_y);
        return (ImgChange::Changes(changes), None);
    }

    let bars = [
        (t.start_x, t.stop_x, t.start_y, t.start_y),
        (t.start_x, t.start_x, t.start_y + 1, t.stop_y),
        (t.start_x + 1, t.stop_x, t.stop_y, t.stop_y),
        (t.stop_x, t.stop_x, t.start_y + 1, t.stop_y - 1),
    ];

    for &(start_x, stop_x, start_y, stop_y) in bars.iter() {
        let (mut changes, split) = compute_for_range(&t, start_x, stop_x, start_y, stop_y);
        all_modifs.append(&mut changes);
        should_split |= split;
    }

    if should_split {
        // We need to split
        // TODO: Try to find a better spot to split at
        let middle_x = (t.stop_x + t.start_x) >> 1;
        let middle_y = (t.stop_y + t.start_y) >> 1;

        let frames = [
            Task { start_x: t.start_x + 1, stop_x: middle_x, start_y: t.start_y + 1, stop_y: middle_y, ..t },
            Task { start_x: middle_x + 1, stop_x: t.stop_x - 1, start_y: t.start_y + 1, stop_y: middle_y, ..t },
            Task { start_x: t.start_x + 1, stop_x: middle_x, start_y: middle_y + 1, stop_y: t.stop_y - 1, ..t },
            Task { start_x: middle_x + 1, stop_x: t.stop_x - 1, start_y: middle_y + 1, stop_y: t.stop_y - 1, ..t }
        ];
        (ImgChange::Changes(all_modifs), Some(frames))
    } else {
        (ImgChange::Changes(all_modifs), None)
    }
}

fn compute_for_range<'a>(
    t: &Task,
    start_x: u32,
    stop_x: u32,
    start_y: u32,
    stop_y: u32,
) -> (Vec<(u32, u32, Option<u64>)>, bool) {
    let mut should_split = false;
    let mut returned: Vec<_> =
        Vec::with_capacity(((stop_x - start_x + 1) * (stop_y - start_y + 1)) as usize);
    for i in start_x..stop_x + 1 {
        for j in start_y..stop_y + 1 {
            let c = complex_from_pos(&t.top_left, &t.bot_right, i, j, t.glob_width, t.glob_height);

            let div = divergent_iteration(&c, t.iter_limit);
            returned.push((i, j, div));

            should_split |= div.is_some();
        }
    }

    (returned, should_split)
}

fn complex_from_pos(
    top_left: &Complex64,
    bot_right: &Complex64,
    i: u32,
    j: u32,
    width: u32,
    height: u32,
) -> Complex64 {
    let x_range = top_left.re - bot_right.re;
    let y_range = top_left.im - bot_right.im;
    let x = (i as f64 / width as f64) * x_range;
    let y = (j as f64 / height as f64) * y_range;

    Complex64::new(bot_right.re + x, bot_right.im + y)
}

fn divergent_iteration(c: &Complex64, limit: u64) -> Option<u64> {
    let mut z: Complex64 = *c;
    let mut i = 0;

    while z.norm_sqr() < 4.0 && i < limit {
        i += 1;
        z = z * z + c;
    }

    if i < limit {
        Some(i)
    } else {
        None
    }
}

fn gradient(source: &[u8; 3], dest: &[u8; 3], iter_count: u64, modulus: u64) -> [u8; 3] {
    let r_change = dest[0] as i16 - source[0] as i16;
    let g_change = dest[1] as i16 - source[1] as i16;
    let b_change = dest[2] as i16 - source[2] as i16;

    if iter_count == 0 {
        return source.to_owned();
    }

    let percent = (((iter_count.log2() as u64) % modulus) as f64) / (modulus as f64);

    let r = source[0] as i16 + (r_change as f64 * percent) as i16;
    let g = source[1] as i16 + (g_change as f64 * percent) as i16;
    let b = source[2] as i16 + (b_change as f64 * percent) as i16;

    [r as u8, g as u8, b as u8]
}
