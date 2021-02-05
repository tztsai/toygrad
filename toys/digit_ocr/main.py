import http.server
import json
import numpy as np
import os
from subprocess import Popen
from nn import NN, onehot

HOST_NAME = 'localhost'
PORT_NUMBER = 8000
PORT_NUMBER2 = 9876
MODEL_FILE = 'ocr.pkl'

try:
    nn = NN.read(MODEL_FILE)
except FileNotFoundError:
    from setup_nn import nn


def nn_predict(img):
    output = nn.predict(img)
    answer = np.argmax(output)
    return answer


def nn_train(img, lb):
    target = onehot(lb, 10)
    nn.fit(img, target, epochs=10)


class JSONHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(s):
        response_code = 200
        response = ""
        var_len = int(s.headers.get('Content-Length'))
        content = s.rfile.read(var_len);
        payload = json.loads(content);

        if payload.get('train'):
            nn_train(payload['images'], payload['labels'])
        elif payload.get('predict'):
            try:
                response = {"type": "test",
                            "result": str(nn_predict(payload['image']))}
            except:
                response_code = 500
        else:
            response_code = 400

        s.send_response(response_code)
        s.send_header("Content-type", "application/json")
        s.send_header("Access-Control-Allow-Origin", "*")
        s.end_headers()
        if response:
            s.wfile.write(json.dumps(response).encode())
        return

if __name__ == '__main__':
    server_class = http.server.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)
    
    ocr_url = 'http://localhost:%d' % PORT_NUMBER2
    print('Open %s and then open "ocr.html" to play!\n' % ocr_url)
    Popen('python -m http.server %d' % PORT_NUMBER2)
    os.system('start ' + ocr_url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    else:
        print("Unexpected server exception occurred.")
    finally:
        httpd.server_close()
