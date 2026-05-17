import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../models/message_model.dart';

/// Backend (Python) sunucusu ile iletişim kuran servis sınıfıdır.
class ApiService {
  
  /// Platforma göre (Android emülatör, iOS veya Web) backend adresini belirler.
  static String get baseUrl {
    if (kIsWeb) {
      return 'http://localhost:8000'; // Web için
    } else if (Platform.isAndroid) {
      // Android emülatörü bilgisayarın localhost'una bu özel IP ile erişebilir.
      return 'http://10.0.2.2:8000'; 
    } else {
      // iOS simülatörü ve masaüstü (Windows/macOS) uygulamaları için.
      return 'http://localhost:8000';
    }
  }

  /// Tek seferde tam cevabı almak için kullanılan klasik HTTP POST isteği.
  Future<Message> sendMessage(String text) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/chat'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'message': text}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return Message(
          text: data['response'] as String? ?? 'Cevap alınamadı.',
          isUser: false,
          timestamp: DateTime.now(),
        );
      } else {
        return Message(
          text: 'Hata: Sunucu ${response.statusCode} hatası döndürdü.',
          isUser: false,
          timestamp: DateTime.now(),
        );
      }
    } catch (e) {
      return Message(
        text: 'Hata: Sunucuya bağlanılamadı. ($e)',
        isUser: false,
        timestamp: DateTime.now(),
      );
    }
  }

  /// Yapay zekadan cevabı kelime kelime (akış halinde) almak için kullanılır.
  /// Bir Stream döndürür, bu sayede Flutter ekranı her yeni kelimede güncellenebilir.
  Stream<String> sendMessageStream(String text) async* {
    try {
      // Stream isteği oluşturuyoruz
      final request = http.Request('POST', Uri.parse('$baseUrl/chat/stream'));
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode({'message': text});

      // İsteği gönderiyoruz
      final response = await request.send();

      if (response.statusCode == 200) {
        // Gelen veri baytlarını (byte) metne çeviriyoruz ve satır satır okuyoruz.
        yield* response.stream
            .transform(utf8.decoder)
            .transform(const LineSplitter());
      } else {
        yield 'Hata: Sunucu ${response.statusCode} döndürdü.';
      }
    } catch (e) {
      yield 'Hata: Sunucuya bağlanılamadı. ($e)';
    }
  }
}
