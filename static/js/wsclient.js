function wsClient() {
    var self = this

    self.IsOpened = false

    self.open = function (url) {
        self.url = url

        self.ws = new WebSocket(url)
        self.ws.onopen = function (event) {
            self.IsOpened = true;
            self.on('open', event);
        };

        self.ws.onclose = function (event) {
            self.IsOpened = false
            setTimeout(function () {
                self.open(url)
            }, 1000)
        };

        self.ws.onmessage = function (event) {
            self.on('message', event)
        };

        self.ws.onerror = function (event) {
            self.on('error', event)
        };
    };

    self.on = function (e, event) {
        if (self["on" + e] !== undefined && typeof self["on" + e] === 'function') {
            self["on" + e](event)
        }
    };

    self.sendJSON = function (obj) {
        self.ws.send(JSON.stringify(obj));
    };

    return self
}