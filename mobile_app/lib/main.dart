import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Image Denoising',
      theme: ThemeData(useMaterial3: true),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _selectedImage;
  Uint8List? _denoisedBytes;
  bool _isLoading = false;
  bool _sharpen = false;
  String _error = "";

  final ImagePicker picker = ImagePicker();

  final String baseUrl = Platform.isAndroid
      ? "http://10.0.2.2:5000"
      : "http://127.0.0.1:5000";

  Future<void> pickImage() async {
    final XFile? img = await picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 1024,
      maxHeight: 1024,
      imageQuality: 90,
    );

    if (img == null) return;

    setState(() {
      _selectedImage = File(img.path);
      _denoisedBytes = null;
      _error = "";
    });
  }

  Future<void> denoiseImage() async {
    if (_selectedImage == null) {
      setState(() => _error = "Please select an image first");
      return;
    }

    setState(() {
      _isLoading = true;
      _error = "";
    });

    try {
      final uri = Uri.parse(
        "$baseUrl/denoise?sharpen=${_sharpen ? 1 : 0}",
      );

      final request = http.MultipartRequest("POST", uri);
      request.files.add(
        await http.MultipartFile.fromPath("image", _selectedImage!.path),
      );

      final response = await request.send();
      final bytes = await response.stream.toBytes();

      if (response.statusCode == 200) {
        setState(() => _denoisedBytes = bytes);
      } else {
        setState(() => _error = "Server error ${response.statusCode}");
      }
    } catch (e) {
      setState(() => _error = e.toString());
    }

    setState(() => _isLoading = false);
  }

  Future<void> saveOutput() async {
    if (_denoisedBytes == null) return;

    final dir = await getDownloadsDirectory();
    final file = File(
        "${dir!.path}/denoised_${DateTime.now().millisecondsSinceEpoch}.png");

    await file.writeAsBytes(_denoisedBytes!);

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("Saved to ${file.path}")),
    );
  }

  void openFullScreen(ImageProvider image, String title) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => FullImagePage(image: image, title: title),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Image Denoising Autoencoder"),
        backgroundColor: Colors.blueGrey.shade900,
        foregroundColor: Colors.white,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Expanded(
              child: Row(
                children: [
                  Expanded(
                    child: ImageCard(
                      title: "Original",
                      image: _selectedImage == null
                          ? null
                          : FileImage(_selectedImage!),
                      onTap: _selectedImage == null
                          ? null
                          : () => openFullScreen(
                                FileImage(_selectedImage!),
                                "Original Image",
                              ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: ImageCard(
                      title: "Denoised",
                      image: _denoisedBytes == null
                          ? null
                          : MemoryImage(_denoisedBytes!),
                      loading: _isLoading,
                      onTap: _denoisedBytes == null
                          ? null
                          : () => openFullScreen(
                                MemoryImage(_denoisedBytes!),
                                "Denoised Image",
                              ),
                    ),
                  ),
                ],
              ),
            ),

            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Switch(
                  value: _sharpen,
                  onChanged: (v) => setState(() => _sharpen = v),
                ),
                const Text("Sharpen output (optional)"),
              ],
            ),

            if (_error.isNotEmpty)
              Text(_error, style: const TextStyle(color: Colors.red)),

            const SizedBox(height: 8),

            ElevatedButton.icon(
              onPressed: pickImage,
              icon: const Icon(Icons.photo),
              label: const Text("Pick Image"),
            ),

            const SizedBox(height: 6),

            ElevatedButton.icon(
              onPressed: _isLoading ? null : denoiseImage,
              icon: const Icon(Icons.auto_fix_high),
              label: Text(_isLoading ? "Processing..." : "Denoise"),
            ),

            const SizedBox(height: 6),

            ElevatedButton.icon(
              onPressed: _denoisedBytes == null ? null : saveOutput,
              icon: const Icon(Icons.save),
              label: const Text("Save Output"),
            ),
          ],
        ),
      ),
    );
  }
}

class ImageCard extends StatelessWidget {
  final String title;
  final ImageProvider? image;
  final VoidCallback? onTap;
  final bool loading;

  const ImageCard({
    super.key,
    required this.title,
    this.image,
    this.onTap,
    this.loading = false,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: InkWell(
        onTap: image == null ? null : onTap,
        child: Column(
          children: [
            Text(title,
                style:
                    const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            Expanded(
              child: loading
                  ? const Center(child: CircularProgressIndicator())
                  : image == null
                      ? const Center(child: Text("No image"))
                      : Image(image: image!, fit: BoxFit.contain),
            ),
          ],
        ),
      ),
    );
  }
}

class FullImagePage extends StatelessWidget {
  final ImageProvider image;
  final String title;

  const FullImagePage({super.key, required this.image, required this.title});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(title)),
      backgroundColor: Colors.black,
      body: Center(
        child: InteractiveViewer(
          child: Image(image: image),
        ),
      ),
    );
  }
}
