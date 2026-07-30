use noperson::io::VirtualCamera;

fn main() -> anyhow::Result<()> {
    let (w, h) = (640u32, 480u32);
    println!("Opening /dev/video10...");
    let mut vcam = VirtualCamera::open(10, w, h, 30)?;
    println!("Opened: {}", vcam.device_path());

    // Send a solid green frame [R, G, B] per pixel
    let frame_size = (w * h * 3) as usize;
    let mut green_frame = Vec::with_capacity(frame_size);
    for _ in 0..(w * h) {
        green_frame.push(0); // R
        green_frame.push(255); // G
        green_frame.push(0); // B
    }

    println!("Sending green frame...");
    vcam.send_frame(&green_frame)?;
    println!("Frame sent! Check: ffplay /dev/video10 or OBS Video Capture Device");

    // Keep alive for 3 seconds
    std::thread::sleep(std::time::Duration::from_secs(3));
    println!("Done.");
    Ok(())
}
