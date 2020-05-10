use crossbeam_channel::unbounded;
use image::{ImageBuffer, ImageFormat, RgbImage};
use num_complex::Complex64;
use std::sync::mpsc::channel;
use std::thread;
use structopt::StructOpt;
use std::path::PathBuf;
use std::num::ParseFloatError;
use std::str::FromStr;

const BLACK_RGB: image::Rgb<u8> = image::Rgb([0, 0, 0]);

/// Mandelbrot set image generator.
#[derive(StructOpt, Debug)]
#[structopt(name="mandelbrust")]
struct Opt {

	/// Path of the output file
	#[structopt(name = "FILE", parse(from_os_str))]
	file: PathBuf,

	/// Iteration limit, used to determine wether a point converges.
	#[structopt(short="l", long, default_value = "500")]
	iter_limit: u64,

	/// width of the generated image.
	#[structopt(long, default_value = "500")]
	width: u32,

	/// height of the generated image.
	#[structopt(long, default_value = "500")]
	height: u32,

	/// Number of threads to use.
	#[structopt(long, default_value = "4")]
	threads: usize,

	/// The center of the image
	#[structopt(long, default_value = "0,0", parse(try_from_str=parse_complex))]
	center: Complex64,

	/// The span of the image
	#[structopt(long, default_value = "2")]
	span: f64,
}

enum ImgChange {
	Changes(Vec<(u32, u32, image::Rgb<u8>)>),
	Fill(u32, u32, u32, u32),
}

fn parse_complex(s: &str) -> Result<Complex64, ParseFloatError> {

	if let Some(div) = s.find(',') {

		let re = f64::from_str(&s[..div])?;
		let im = f64::from_str(&s[div+1..])?;
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
}

fn main() {
	let opt = Opt::from_args();
	let mut img: RgbImage = ImageBuffer::new(opt.width, opt.height);
	let (work_tx, work_rx) = unbounded();
	let (img_tx, img_rx) = channel();

	let top_left = opt.center + Complex64::new(opt.span, opt.span);
	let bot_right = opt.center + Complex64::new(-opt.span, -opt.span);

	work_tx
		.send(Some(Task {
			start_x: 0,
			start_y: 0,
			stop_x: opt.width,
			stop_y: opt.height,
			top_left,
			bot_right,
			glob_width: opt.width,
			glob_height: opt.height,
			iter_limit: opt.iter_limit
		}))
		.unwrap();

	// let work = Arc::new(Mutex::new(work));

	let mut threads: Vec<_> = vec![];

	for _ in 0..opt.threads {
		let w_tx = work_tx.clone();
		let w_rx = work_rx.clone();
		let out = img_tx.clone();
		threads.push(thread::spawn(move || {
			while let Some(task) = w_rx.recv().unwrap() {
				let start_x = task.start_x;
				let start_y = task.start_y;
				let stop_x = task.stop_x;
				let stop_y = task.stop_y;

				let (modification, subtasks) = process_task(task);

				out.send(modification).unwrap();

				if let Some((t1, t2, t3, t4)) = subtasks {
					w_tx.send(Some(t1)).unwrap();
					w_tx.send(Some(t2)).unwrap();
					w_tx.send(Some(t3)).unwrap();
					w_tx.send(Some(t4)).unwrap();
				} else {
					out.send(ImgChange::Fill(
						start_x + 1,
						start_y + 1,
						stop_x - 1,
						stop_y - 1,
					))
					.unwrap();
				}
			}
		}));
	}

	let mut pixels_drawn: u64 = 0;

	while pixels_drawn < (opt.width as u64 * opt.height as u64) {
		match img_rx.recv().unwrap() {
			ImgChange::Changes(changes) => {
				for (x, y, _color) in changes {
					let image::Rgb(color) = img.get_pixel(x, y);
					let new_color = image::Rgb([ color[0]+30, color[0]+30, color[0]+30 ]);

					if color[0] == 0 {
						pixels_drawn += 1;
					}
					img.put_pixel(x, y, new_color);
				}
			}
			ImgChange::Fill(start_x, start_y, stop_x, stop_y) => {
				for i in start_x..stop_x {
					for j in start_y..stop_y {
						img.put_pixel(i, j, BLACK_RGB);
						pixels_drawn += 1;
					}
				}
			}
		}
	}

	for _ in 0..opt.threads {
		work_tx.send(None).unwrap();
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

fn process_task(t: Task) -> (ImgChange, Option<(Task, Task, Task, Task)>) {
	let mut top = compute_for_range(&t, t.start_x, t.stop_x, t.start_y, t.start_y + 1);
	let mut right = compute_for_range(&t, t.stop_x - 1, t.stop_x, t.start_y + 1, t.stop_y);
	let mut bot = compute_for_range(&t, t.start_x, t.stop_x - 1, t.stop_y - 1, t.stop_y);
	let mut left = compute_for_range(&t, t.start_x, t.start_x + 1, t.start_y + 1, t.stop_y - 1);

	let mut all_modifs = Vec::new();
	all_modifs.append(&mut top.0);
	all_modifs.append(&mut right.0);
	all_modifs.append(&mut bot.0);
	all_modifs.append(&mut left.0);

	// Compute surrouding rectangle
	if top.1 || right.1 || bot.1 || left.1 {
		// We need to split
		// TODO: Try to find a better spot to split at
		let middle_x = (t.stop_x + t.start_x) / 2;
		let middle_y = (t.stop_y + t.start_y) / 2;

		let top_left = Task {
			start_x: t.start_x + 1,
			start_y: t.start_y + 1,
			stop_x: middle_x,
			stop_y: middle_y,
			..t
		};

		let top_right = Task {
			start_x: middle_x,
			start_y: t.start_y + 1,
			stop_x: t.stop_x - 1,
			stop_y: middle_y,
			..t
		};
		let bot_left = Task {
			start_x: t.start_x + 1,
			start_y: middle_y,
			stop_x: middle_x,
			stop_y: t.stop_y - 1,
			..t
		};
		let bot_right = Task {
			start_x: middle_x,
			start_y: middle_y,
			stop_x: t.stop_x - 1,
			stop_y: t.stop_y - 1,
			..t
		};
		(
			ImgChange::Changes(all_modifs),
			Some((top_left, top_right, bot_left, bot_right)),
		)
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
) -> (Vec<(u32, u32, image::Rgb<u8>)>, bool) {
	let mut should_split = false;
	let mut returned: Vec<_> = Vec::new();
	for i in start_x..stop_x {
		for j in start_y..stop_y {
			let c = complex_from_pos(t.top_left, t.bot_right, i, j, t.glob_width, t.glob_height);

			if let Some(div) = divergent_iteration(c, t.iter_limit) {
				should_split = true;
				returned.push((
					i,
					j,
					image::Rgb(gradient([139, 233, 253], [68, 71, 90], div, t.iter_limit / 10)),
				));
			} else {
				returned.push((i, j, BLACK_RGB));
			}
		}
	}

	(returned, should_split)
}

fn complex_from_pos(
	top_left: Complex64,
	bot_right: Complex64,
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

fn divergent_iteration(c: Complex64, limit: u64) -> Option<u64> {
	let mut z: Complex64 = Complex64::new(0., 0.);
	for i in 0..limit {
		z = z * z + c;

		if z.norm_sqr() > 4.0 {
			return Some(i);
		}
	}

	None
}

fn gradient(source: [u8; 3], dest: [u8; 3], index: u64, modulus: u64) -> [u8; 3] {
	let r_change = dest[0] as i16 - source[0] as i16;
	let g_change = dest[1] as i16 - source[1] as i16;
	let b_change = dest[2] as i16 - source[2] as i16;

	let mut percent = 2. * (index % ( 2 * modulus)) as f64 / (2 * modulus) as f64 - 1.;
	percent = percent * percent;

	let r = source[0] as i16 + (r_change as f64 * percent) as i16;
	let g = source[1] as i16 + (g_change as f64 * percent) as i16;
	let b = source[2] as i16 + (b_change as f64 * percent) as i16;

	[r as u8, g as u8, b as u8]
}
