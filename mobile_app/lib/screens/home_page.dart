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
  bool loading = false;

  double strength = 1.0;
  bool sharpen = false;

  String psnr = "";
  String ssim = "";
  String mse = "";

  // If Android phone: replace with IPv4 (ipconfig)
  final String api = "http://127.0.0.1:5000/denoise";

  Future pickImage() async {
    final img = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (img != null) {
      setState(() {
        selectedImage = File(img.path);
        denoisedImage = null;
        psnr = ssim = mse = "";
      });
    }
  }

  Future runDenoise() async {
    if (selectedImage == null) return;

    setState(() => loading = true);

    final b64 = base64Encode(selectedImage!.readAsBytesSync());
    final resp = await http.post(
      Uri.parse(api),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "image": b64,
        "strength": strength,
        "sharpen": sharpen,
      }),
    );

    final data = jsonDecode(resp.body);

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

            if (selectedImage != null && denoisedImage != null)
              Container(
                height: 260,
                margin: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(15),
                  color: Colors.white,
                  boxShadow: [BoxShadow(blurRadius: 10, color: Colors.black26)],
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(15),
                  child: BeforeAfter(
                    before: Image.file(selectedImage!, fit: BoxFit.cover),
                    after: Image.memory(denoisedImage!, fit: BoxFit.cover),
                  ),
                ),
              ),

            if (selectedImage != null && denoisedImage == null)
              Image.file(selectedImage!, height: 260, fit: BoxFit.cover),

            const SizedBox(height: 20),

            Text("Denoise Strength: ${(strength * 100).toInt()}%"),
            Slider(
              value: strength,
              min: 0,
              max: 1,
              divisions: 100,
              label: "${(strength * 100).toInt()}%",
              onChanged: (v) => setState(() => strength = v),
            ),

            SwitchListTile(
              title: const Text("Sharpen Output"),
              value: sharpen,
              onChanged: (v) => setState(() => sharpen = v),
            ),

            if (loading) const CircularProgressIndicator(),

            if (denoisedImage != null)
              Container(
                margin: const EdgeInsets.all(12),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [BoxShadow(color: Colors.black26, blurRadius: 6)],
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
