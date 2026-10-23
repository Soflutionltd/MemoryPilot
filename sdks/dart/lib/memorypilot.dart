import 'dart:convert';

import 'package:http/http.dart' as http;

class MemoryPilotException implements Exception {
  MemoryPilotException(this.status, this.body);

  final int status;
  final Object? body;

  @override
  String toString() {
    if (body is Map && (body as Map)['error'] is String) {
      return (body as Map)['error'] as String;
    }
    return 'MemoryPilot HTTP $status';
  }
}

class MemoryPilot {
  MemoryPilot({
    required String baseUrl,
    required this.apiKey,
    this.subject,
    http.Client? httpClient,
  }) : baseUrl = baseUrl.replaceAll(RegExp(r'/$'), ''),
       _http = httpClient ?? http.Client();

  final String baseUrl;
  final String apiKey;
  final String? subject;
  final http.Client _http;

  MemoryPilot forSubject(String subject) {
    return MemoryPilot(
      baseUrl: baseUrl,
      apiKey: apiKey,
      subject: subject,
      httpClient: _http,
    );
  }

  Future<Map<String, Object?>> health() {
    return _send('GET', '/health', auth: false);
  }

  Future<Map<String, Object?>> addMemory(Map<String, Object?> memory) {
    return _send('POST', '/v1/memories', body: memory);
  }

  Future<Map<String, Object?>> listMemories({
    String? project,
    String? kind,
    int? limit,
    int? offset,
  }) {
    final query = <String, String>{
      if (project != null) 'project': project,
      if (kind != null) 'kind': kind,
      if (limit != null) 'limit': '$limit',
      if (offset != null) 'offset': '$offset',
    };
    final suffix = query.isEmpty ? '' : '?${Uri(queryParameters: query).query}';
    return _send('GET', '/v1/memories$suffix');
  }

  Future<Map<String, Object?>> getMemory(String id) {
    return _send('GET', '/v1/memories/${Uri.encodeComponent(id)}');
  }

  Future<Map<String, Object?>> updateMemory(
    String id,
    Map<String, Object?> changes,
  ) {
    return _send('PATCH', '/v1/memories/${Uri.encodeComponent(id)}', body: changes);
  }

  Future<Map<String, Object?>> deleteMemory(String id) {
    return _send('DELETE', '/v1/memories/${Uri.encodeComponent(id)}');
  }

  Future<Map<String, Object?>> search(Map<String, Object?> query) {
    return _send('POST', '/v1/search', body: query);
  }

  Future<Map<String, Object?>> exportSubject() {
    return _send('GET', '/v1/subject/export');
  }

  Future<Map<String, Object?>> deleteSubject() {
    return _send('DELETE', '/v1/subject');
  }

  Future<Map<String, Object?>> _send(
    String method,
    String path, {
    Map<String, Object?>? body,
    bool auth = true,
  }) async {
    final headers = <String, String>{'accept': 'application/json'};
    if (auth) {
      headers['authorization'] = 'Bearer $apiKey';
      if (subject != null) {
        headers['x-memorypilot-subject'] = subject!;
      }
    }
    http.Response response;
    final uri = Uri.parse('$baseUrl$path');
    final encoded = body == null ? null : jsonEncode(body);
    if (encoded != null) {
      headers['content-type'] = 'application/json';
    }
    switch (method) {
      case 'GET':
        response = await _http.get(uri, headers: headers);
      case 'POST':
        response = await _http.post(uri, headers: headers, body: encoded);
      case 'PATCH':
        response = await _http.patch(uri, headers: headers, body: encoded);
      case 'DELETE':
        response = await _http.delete(uri, headers: headers, body: encoded);
      default:
        throw ArgumentError.value(method, 'method');
    }
    final decoded = response.body.isEmpty
        ? <String, Object?>{}
        : jsonDecode(response.body) as Object?;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw MemoryPilotException(response.statusCode, decoded);
    }
    if (decoded is Map<String, Object?>) {
      return decoded;
    }
    if (decoded is Map) {
      return decoded.map((key, value) => MapEntry('$key', value));
    }
    return {'value': decoded};
  }
}
