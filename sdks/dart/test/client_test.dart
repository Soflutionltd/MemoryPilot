import 'dart:convert';
import 'dart:io';

import 'package:memorypilot/memorypilot.dart';
import 'package:test/test.dart';

void main() {
  test('adds a memory for an end user and surfaces HTTP errors', () async {
    final seen = <Map<String, String>>[];
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    server.listen((request) async {
      final body = await utf8.decodeStream(request);
      seen.add({
        'path': request.uri.path,
        'authorization': request.headers.value('authorization') ?? '',
        'subject': request.headers.value('x-memorypilot-subject') ?? '',
        'body': body,
      });
      request.response.headers.contentType = ContentType.json;
      if (request.uri.path == '/v1/memories') {
        request.response.statusCode = 201;
        request.response.write(
          jsonEncode({
            'memory': {'id': 'm1'},
            'merged': false,
          }),
        );
      } else {
        request.response.statusCode = 403;
        request.response.write(
          jsonEncode({'error': "This key lacks the 'read' permission"}),
        );
      }
      await request.response.close();
    });

    final client = MemoryPilot(
      baseUrl: 'http://127.0.0.1:${server.port}',
      apiKey: 'mpk_test',
    ).forSubject('alice');
    final created = await client.addMemory({'content': 'Alice prefers green tea'});
    expect(created['memory'], containsPair('id', 'm1'));
    expect(seen.first['authorization'], 'Bearer mpk_test');
    expect(seen.first['subject'], 'alice');
    await expectLater(
      client.search({'query': 'tea'}),
      throwsA(
        isA<MemoryPilotException>().having((error) => error.status, 'status', 403),
      ),
    );
    await server.close(force: true);
  });
}
