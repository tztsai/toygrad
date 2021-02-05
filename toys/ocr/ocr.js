// Canvas shape
const CANVAS_WIDTH = 420
const TRANSLATED_WIDTH = 28
const PIXEL_WIDTH = 15 // TRANSLATED_WIDTH = CANVAS_WIDTH / PIXEL_WIDTH

// Training variables
const BATCH_SIZE = 2

// Server Variables
const PORT = "8000"
const HOST = "http://localhost"

// Colors
const BLACK = "#000000"
const BLUE = "#0000ff"
const YELLOW = "#FFFF00"
const DARK_YELLOW = "#888844"


var ocrDemo = {
    trainArrays: [],
    trainLabels: [],
    semiTransparent: new Set(),
    trainingRequestCount: 0,

    onLoadFunction: function() {
        this.resetCanvas();
    },

    resetCanvas: function() {
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');

        this.data = [];
        this.semiTransparent.clear();

        digit = document.getElementById("digit");
        digit.style.color = BLACK;
        digit.value = "?";

        ctx.fillStyle = BLACK;
        ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_WIDTH);
        var matrixSize = TRANSLATED_WIDTH * TRANSLATED_WIDTH;
        while (matrixSize--) this.data.push(0);
        this.drawGrid(ctx);

        canvas.onmousemove = function(e) { this.onMouseMove(e, ctx, canvas) }.bind(this);
        canvas.onmousedown = function(e) { this.onMouseDown(e, ctx, canvas) }.bind(this);
        canvas.onmouseup = function(e) { this.onMouseUp(e, ctx) }.bind(this);
    },

    drawGrid: function(ctx) {
        for (var x = PIXEL_WIDTH, y = PIXEL_WIDTH; x < CANVAS_WIDTH; x += PIXEL_WIDTH, y += PIXEL_WIDTH) {
            ctx.strokeStyle = BLUE;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },

    onMouseMove: function(e, ctx, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        this.drawMark(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseDown: function(e, ctx, canvas) {
        canvas.isDrawing = true;
        this.drawMark(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseUp: function(e) {
        canvas.isDrawing = false;
    },

    fillPixel: function(ctx, x, y, color=YELLOW) {
        if (color != YELLOW) {
            let entry = x + ' ' + y;
            if (this.semiTransparent.has(entry)) {
                color = YELLOW;
            } else {
                this.semiTransparent.add(entry);
            }
        }
        ctx.fillStyle = color;
        ctx.fillRect(x * PIXEL_WIDTH, y * PIXEL_WIDTH, PIXEL_WIDTH, PIXEL_WIDTH);

        let value = (color == YELLOW)? 1 : 0.5;
        this.data[((y - 1)  * TRANSLATED_WIDTH + x) - 1] = value;
    },

    drawMark: function(ctx, x, y) {
        var xPixel = Math.floor(x / PIXEL_WIDTH);
        var yPixel = Math.floor(y / PIXEL_WIDTH);
        this.fillPixel(ctx, xPixel, yPixel);

        // also fill surrounding squares
        ctx.fillStyle = '#666666';
        var i = 0;
        [xPixel-1, xPixel+1, yPixel-1, yPixel+1].forEach(z => {
            let x = (i < 2)? z : xPixel;
            let y = (i < 2)? yPixel : z;
            if (z >= 0 && z < TRANSLATED_WIDTH) {
                this.fillPixel(ctx, x, y, DARK_YELLOW);
            }
            i += 1;
        });
    },

    train: function() {
        var digitVal = document.getElementById("digit").value;
        if (!digitVal || this.data.indexOf(1) < 0) {
            alert("Please type and draw a digit value in order to train the network");
            return;
        }
        this.trainArrays.push(this.data);
        this.trainLabels.push(parseInt(digitVal));
        this.trainingRequestCount++;

        // Time to send a training batch to the server.
        if (this.trainingRequestCount == BATCH_SIZE) {
            alert("Sending training data to server...");
            var json = {
                images: this.trainArrays,
                labels: this.trainLabels,
                train: true
            };

            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];
        }
    },

    test: function() {
        if (this.data.indexOf(1) < 0) {
            alert("Please draw a digit in order to test the network");
            return;
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);
    },

    receiveResponse: function(xmlHttp) {
        if (xmlHttp.status != 200) {
            alert("Server returned status " + xmlHttp.status);
            return;
        }
        var responseJSON = JSON.parse(xmlHttp.responseText);
        if (xmlHttp.responseText && responseJSON.type == "test") {
            let digit = document.getElementById("digit");
            digit.value = responseJSON.result;
            digit.style.color = "brown";
        }
    },

    onError: function(e) {
        alert("Error occurred while connecting to server: " + e.target.statusText);
    },

    sendData: function(json) {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open('POST', HOST + ":" + PORT, false);
        xmlHttp.onload = function() { this.receiveResponse(xmlHttp); }.bind(this);
        xmlHttp.onerror = function() { this.onError(xmlHttp) }.bind(this);
        var msg = JSON.stringify(json);
        xmlHttp.setRequestHeader('Content-length', msg.length);
        xmlHttp.setRequestHeader("Connection", "close");
        xmlHttp.send(msg);
    }
}