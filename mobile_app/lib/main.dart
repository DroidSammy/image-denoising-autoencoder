import 'package:flutter/material.dart';
import 'screens/home_page.dart';

void main() {
  runApp(const ImageDenoiseApp());
}

class ImageDenoiseApp extends StatelessWidget {
  const ImageDenoiseApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: "Image Denoising Autoencoder",
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.purple),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}
