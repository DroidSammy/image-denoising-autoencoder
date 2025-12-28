import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:before_after/before_after.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? selectedImage;
  Uint8List? denoisedImage;

  double strength = 1.0;
  bool sharpen = false;
  bool loading = false;

  // 👇 THIS MAKES THE SLIDER MOVABLE
  double sliderPosition = 0.5;

  // 📌 CHANGE ONLY IF USING ANDROID (use IPv4 from ipconfig)
  final String api = "http://127.0.0.1:5000/denoise";

  String psnr = "", ssim = "", mse = "";

  Future pickImage() async {
    final img = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (img == null) return;
    setState(() {
      selectedImage = File(img.path);
      denoisedImage = null;
      psnr = ssim = mse = "";
      sliderPosition = 0.5;
    });
  }

  Future runDenoise() async {
    if (selectedImage == null) return;
    setState(() => loading = true);

    final bytes = await selectedImage!.readAsBytes();
    final response = await http.post(
      Uri.parse(api),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "image": base64Encode(bytes),
        "strength": strength,
        "sharpen": sharpen
      }),
    );

    final data = jsonDecode(response.body);
    setState(() {
      denoisedImage = base64Decode(data["denoised_image"]);
      psnr = data["metrics"]["psnr"].toString();
      ssim = data["metrics"]["ssim"].toString();
      mse = data["metrics"]["mse"].toString();
      loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xfff6eaff),
      appBar: AppBar(
        backgroundColor: Colors.purple,
        title: const Text("Image Denoising Autoencoder",
            style: TextStyle(color: Colors.white)),
      ),

      body: SingleChildScrollView(
        child: Column(
          children: [
            const SizedBox(height: 20),

            // -------------- BEFORE/AFTER WIDGET (SLIDER FIXED & MOVABLE) ------------
            if (selectedImage != null && denoisedImage != null)
              Container(
                height: 260,
                margin: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(15),
                  color: Colors.white,
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(15),
                  child: BeforeAfter(
                    before: Image.file(selectedImage!, fit: BoxFit.cover),
                    after: Image.memory(denoisedImage!, fit: BoxFit.cover),

                    // 👇 THIS MAKES THE DIVIDER SLIDE
                    value: sliderPosition,
                    onValueChanged: (v) {
                      setState(() => sliderPosition = v);
                    },

                  ),
                ),
              ),

            if (selectedImage != null && denoisedImage == null)
              Image.file(selectedImage!, height: 260, fit: BoxFit.cover),

            const SizedBox(height: 20),

            // ---------------- SLIDER (NO BREAKING PROPERTIES) ----------------
            Text("Denoise Strength: ${(strength * 100).toInt()}%"),
            Slider(
              value: strength,
              min: 0,
              max: 1,
              onChanged: (v) => setState(() => strength = v),
            ),

            // ---------------- SWITCH (WINDOWS COMPATIBLE) ----------------
            SwitchListTile(
              title: const Text("Sharpen Output"),
              value: sharpen,
              onChanged: (v) => setState(() => sharpen = v),
            ),

            if (loading) const CircularProgressIndicator(),

            // ---------------- METRICS DISPLAY ----------------
            if (denoisedImage != null)
              Container(
                padding: const EdgeInsets.all(12),
                margin: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    Text("PSNR: $psnr dB"),
                    Text("SSIM: $ssim"),
                    Text("MSE: $mse"),
                  ],
                ),
              ),

            // ---------------- ACTION BUTTONS ----------------
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: pickImage,
                  icon: const Icon(Icons.photo),
                  label: const Text("Pick Image"),
                ),
                ElevatedButton.icon(
                  onPressed: runDenoise,
                  icon: const Icon(Icons.auto_fix_high),
                  label: const Text("Denoise"),
                ),
              ],
            ),

            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }
}
