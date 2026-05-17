import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:flutter_math_fork/flutter_math.dart';
import 'package:markdown/markdown.dart' as md;
import '../models/message_model.dart';
import '../services/api_service.dart';

/// Ana sohbet ekranı
class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  // Yazı giriş alanını kontrol eden controller
  final TextEditingController _controller = TextEditingController();
  // Mesaj listesi (Sohbet geçmişi burada tutulur)
  final List<Message> _messages = [];
  // Backend ile konuşan servis
  final ApiService _apiService = ApiService();
  // Yanıt beklenirken yükleme göstergesini kontrol eder
  bool _isLoading = false;

  /// Gönder butonuna basıldığında çalışan ana fonksiyon
  void _handleSend() async {
    if (_controller.text.trim().isEmpty) return;

    final input = _controller.text;
    
    // 1. Kullanıcının mesajını oluştur ve listeye ekle
    final userMessage = Message(
      text: input,
      isUser: true,
      timestamp: DateTime.now(),
    );

    setState(() {
      _messages.add(userMessage);
      _isLoading = true; // Yükleniyor animasyonunu başlat
    });

    // Giriş alanını temizle
    _controller.clear();

    // 2. Yapay zeka için boş bir mesaj kabuğu oluştur ve listeye ekle
    // Gelen kelimeleri bu kabuğun içine dolduracağız.
    final aiMessage = Message(
      text: '',
      isUser: false,
      timestamp: DateTime.now(),
    );

    setState(() {
      _messages.add(aiMessage);
    });

    String fullResponse = '';
    bool firstChunk = true;

    try {
      // 3. Backend'den akışlı (streaming) cevabı dinle
      await for (final chunk in _apiService.sendMessageStream(input)) {
        if (firstChunk) {
          // İlk kelime geldiği an yükleme animasyonunu durdur
          setState(() {
            _isLoading = false;
          });
          firstChunk = false;
        }
        
        // Gelen her parçayı ana metne ekle ve ekranı güncelle
        fullResponse += chunk;
        setState(() {
          aiMessage.text = fullResponse;
        });
      }
    } catch (e) {
      setState(() {
        aiMessage.text = 'Bir hata oluştu: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    // Tasarım renkleri
    const navyBlue = Color(0xFF050555);
    const darkNavy = Color(0xFF050555);

    return Scaffold(
      backgroundColor: navyBlue,
      appBar: AppBar(
        title: const Text('TYT MATEMATİK ASİSTANI', 
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        backgroundColor: darkNavy,
        iconTheme: const IconThemeData(color: Colors.white),
        elevation: 0,
      ),
      drawer: _buildDrawer(navyBlue, darkNavy), // Yan menü
      body: Column(
        children: [
          // Mesajların listelendiği alan
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                return _buildMessageBubble(_messages[index]);
              },
            ),
          ),
          // Yükleniyor göstergesi (Eğer aktifse)
          if (_isLoading)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: CircularProgressIndicator(color: Colors.white),
            ),
          // Yazı yazma alanı
          _buildInputArea(),
        ],
      ),
    );
  }

  /// Mesaj Baloncuklarını oluşturan yardımcı widget
  Widget _buildMessageBubble(Message message) {
    final isUser = message.isUser;
    const userBubbleColor = Color(0xFF1A1A6E);
    const aiBubbleColor = Color(0xFF0D0D4B);

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.85),
        decoration: BoxDecoration(
          color: isUser ? userBubbleColor : aiBubbleColor,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(16),
            topRight: const Radius.circular(16),
            bottomLeft: Radius.circular(isUser ? 16 : 0),
            bottomRight: Radius.circular(isUser ? 0 : 16),
          ),
        ),
        child: isUser
            ? Text(message.text, style: const TextStyle(color: Colors.white, fontSize: 16))
            : MarkdownBody(
                data: message.text,
                selectable: true,
                builders: {
                  'latex': LatexElementBuilder(), // Matematiksel ifadeler için özel çizici
                },
                styleSheet: MarkdownStyleSheet(
                  p: const TextStyle(color: Colors.white, fontSize: 16),
                  strong: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                ),
              ),
      ),
    );
  }

  /// Alt kısımdaki yazı yazma alanı
  Widget _buildInputArea() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      color: const Color(0xFF050555),
      child: SafeArea(
        child: Row(
          children: [
            Expanded(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                decoration: BoxDecoration(
                  color: const Color(0xFF161B4D),
                  borderRadius: BorderRadius.circular(24),
                ),
                child: TextField(
                  controller: _controller,
                  style: const TextStyle(color: Colors.white),
                  decoration: const InputDecoration(
                    hintText: 'Mesajınızı yazın...',
                    hintStyle: TextStyle(color: Colors.white54),
                    border: InputBorder.none,
                  ),
                  onSubmitted: (_) => _handleSend(),
                ),
              ),
            ),
            const SizedBox(width: 8),
            CircleAvatar(
              backgroundColor: Colors.white,
              child: IconButton(
                icon: const Icon(Icons.send, color: Color(0xFF050555)),
                onPressed: _handleSend,
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Yan menü (Drawer)
  Widget _buildDrawer(Color navyBlue, Color darkNavy) {
    return Drawer(
      backgroundColor: navyBlue,
      child: Column(
        children: [
          DrawerHeader(
            decoration: BoxDecoration(color: darkNavy),
            child: const Center(
              child: Text(
                'Menü',
                style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
              ),
            ),
          ),
          ListTile(
            leading: const Icon(Icons.history, color: Colors.white),
            title: const Text('Geçmiş Chatler', style: TextStyle(color: Colors.white)),
            onTap: () => Navigator.pop(context),
          ),
          const Spacer(),
          const Padding(
            padding: EdgeInsets.all(16.0),
            child: Text('Versiyon 1.0.0', style: TextStyle(color: Colors.white70)),
          ),
        ],
      ),
    );
  }
}

/// Markdown içinde geçen matematiksel ifadeleri ($...$) yakalayıp 
/// güzel bir şekilde (LaTeX) çizen sınıf.
class LatexElementBuilder extends MarkdownElementBuilder {
  @override
  Widget? visitText(md.Text text, TextStyle? preferredStyle) {
    final String data = text.text;
    if (data.contains(r'$')) {
      final parts = data.split(r'$');
      List<Widget> children = [];
      
      for (var i = 0; i < parts.length; i++) {
        if (i % 2 == 0) {
          // Normal metin
          if (parts[i].isNotEmpty) {
            children.add(Text(parts[i], style: preferredStyle));
          }
        } else {
          // Matematiksel ifade
          children.add(Padding(
            padding: const EdgeInsets.symmetric(horizontal: 2.0),
            child: Math.tex(
              parts[i],
              textStyle: preferredStyle?.copyWith(
                fontSize: (preferredStyle.fontSize ?? 16) + 2,
              ) ?? const TextStyle(fontSize: 18, color: Colors.white),
            ),
          ));
        }
      }
      return Wrap(
        crossAxisAlignment: WrapCrossAlignment.center,
        children: children,
      );
    }
    return Text(data, style: preferredStyle);
  }
}
