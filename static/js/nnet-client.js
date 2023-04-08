function NNetClient(netid) {
    var self = this;

    self.netid = netid;
    self.netboxid = 'net-box-' + netid;
    self.netbox = $('#' + self.netboxid);

    self.wsRecipient = netid;

    self.eventCode = function (code) {
        return netid + "." + code;
    };


    var plot;

    self.BindWsEvents = function (wsListener) {
        self.wsListener = wsListener;

        // self.netbox.find('.network-create').bind('click', function () {

        $('.network-create').bind('click', function () {
            self.wsListener.wsClient.sendJSON({
                code: self.eventCode( "create"),
                data: {}
            });
        });

        // self.netbox.find('.network-load').bind('click', function () {
        $('.network-load').bind('click', function () {
            self.wsListener.wsClient.sendJSON({
                code: self.eventCode( "load"),
                data: {}
            });
        });

        // self.netbox.find('.network-save').bind('click', function () {
        $('.network-save').bind('click', function () {
            self.wsListener.wsClient.sendJSON({
                code: self.eventCode( "save"),
                data: {}
            });
        });

        // self.netbox.find('.training-start').bind('click', function () {
        $('.training-start').bind('click', function () {
            self.wsListener.wsClient.sendJSON({
                code: self.eventCode( "training-start"),
                data: {}
            });
        });

        // self.netbox.find('.training-stop').bind('click', function () {
        $('.training-stop').bind('click', function () {
            self.wsListener.wsClient.sendJSON({
                code: self.eventCode("training-stop"),
                data: {}
            });
        });

        self.wsListener.Bind(self.eventCode("prediction-block"), self.wsRecipient, function (event) {
            let predictionBlock = self.netbox.find('.cnn-prediction-block .panel-body');
            if (predictionBlock.length === 0) {
                return;
            }

            let output = '';
            for (i = 0; i < event.data.length; i++) {
                let predictions = '';
                let predictionEvent = event.data[i]
                if (predictionEvent.predictions.length > 0) {
                    for (let i = 0; i < predictionEvent.predictions.length; i++) {
                        let percent = predictionEvent.predictions[i].percent;
                        let validClass = '';

                        if (predictionEvent.valid) {
                            if (predictionEvent.predictions[i].label === predictionEvent.target) {
                                validClass = 'valid';
                            }
                        } else if (predictionEvent.predictions[i].label === predictionEvent.target) {
                            validClass = 'invalid-target';
                        } else if (predictionEvent.predictions[i].label === predictionEvent.output) {
                            validClass = 'invalid-output';
                        }

                        predictions += '' +
                            '<div class="prediction-bar '+validClass+'">' +
                            '<div class="prediction-bar-progress" style="width: ' + percent + '%"></div>' +
                            '<div class="prediction-bar-label">' +
                            predictionEvent.predictions[i].label + ': ' +
                            predictionEvent.predictions[i].value +
                            '</div>'+
                            '<div class="prediction-bar-percent">' + percent + '%</div>'+
                            '</div>';
                    }
                }

                output += '<table>\n' +
                    '<tr>\n' +
                    '<td valign="top">\n' +
                    '<img class="input-image" src="data:image/png;base64,'+predictionEvent.image+'">\n' +
                    '</td>\n' +
                    '<td valign="top">\n' +
                    '<div class="prediction-bars">'+predictions+'</div>\n' +
                    '</td>\n' +
                    '</tr>\n' +
                    '</table>';
            }

            predictionBlock.html(output + '<div style="clear: both"></div>');
        });
        //
        // self.wsListener.Bind(self.eventCode("prediction-block"), self.wsRecipient, function (event) {
        //     let predictionBlock = self.netbox.find('.cnn-prediction-block');
        //     if (predictionBlock.length === 0) {
        //         return;
        //     }
        //
        //     let inputImage = predictionBlock.find('.input-image');
        //     if (inputImage.length > 0) {
        //         inputImage[0].src = 'data:image/png;base64,' + event.data.image;
        //     }
        //
        //     let predictionBars = predictionBlock.find('.prediction-bars');
        //     if (predictionBars.length > 0) {
        //         let predictions = '';
        //
        //         for (let i = 0; i < event.data.predictions.length; i++) {
        //             let percent = event.data.predictions[i].percent;
        //             let validClass = '';
        //
        //             if (event.data.valid) {
        //                 if (event.data.predictions[i].label === event.data.target) {
        //                     validClass = 'valid';
        //                 }
        //             } else if (event.data.predictions[i].label === event.data.target) {
        //                 validClass = 'invalid-target';
        //             } else if (event.data.predictions[i].label === event.data.output) {
        //                 validClass = 'invalid-output';
        //             }
        //
        //             predictions += '' +
        //                 '<div class="prediction-bar '+validClass+'">' +
        //                 '<div class="prediction-bar-progress" style="width: ' + percent + '%"></div>' +
        //                 '<div class="prediction-bar-label">' +
        //                     event.data.predictions[i].label + ': ' +
        //                     event.data.predictions[i].value +
        //                 '</div>'+
        //                 '<div class="prediction-bar-percent">' + percent + '%</div>'+
        //                 '</div>';
        //         }
        //
        //         predictionBars.html(predictions);
        //     }
        // });

        self.wsListener.Bind(self.eventCode("train-layer-info"), self.wsRecipient, function (event) {
            var box = $("#" + self.netboxid + " .train-layer-info-"+event.data.Index);
            var info = box.find(".info")
            if (info.length === 0) {
                box.append("" +
                    "<div class='panel panel-default'>" +
                    "<div class='panel-heading info'></div>" +
                    "<div class='panel-body'>" +

                    (!event.data.InputGradients ? "" :
                            "<div>input gradients</div>" +
                            "<div class='input-grads'></div>"
                    ) +

                    (!event.data.WeightsGradients ? "" :
                        "<div>weights gradients</div>" +
                        "<div class='weights-grads'></div>"
                    ) +

                    (!event.data.WeightsImages ? "" :
                        "<div>weights images</div>" +
                        "<div class='weights-images'></div>"
                    ) +

                    (!event.data.OutputImages ? "" :
                            "<div>output</div>" +
                            "<div class='output-images'></div>"
                    ) +

                    (!event.data.Weights ? "" :
                            "<div>weights</div>" +
                            "<div class='weights'></div>"
                    ) +

                    "</div>" +
                    "</div>"
                )
                info = box.find(".info")
            }

            var s = ""

            s += "Layer [" + event.data.Index + "] " + event.data.LayerType

            info.html(s)

            if (event.data.InputGradients) {
                var inputGrads = box.find(".input-grads")
                let layer = event.data.Index
                for (var i = 0; i < event.data.InputGradients.length; i++) {
                    let img = inputGrads.find('img.train-input-gradients-'+layer+'-'+i);
                    if (img.length === 0) {
                        inputGrads.append('<img class="train-input-gradients-'+layer+'-'+i+'" width="32" height="32" style="border: solid 1px #fff">');
                        img = inputGrads.find('img.train-input-gradients-'+layer+'-'+i);
                    }

                    img[0].src = 'data:image/png;base64,' + event.data.InputGradients[i];
                }
            }

            if (event.data.WeightsGradients) {
                var weightsGrads = box.find(".weights-grads")
                let layer = event.data.Index
                for (var i = 0; i < event.data.WeightsGradients.length; i++) {
                    let img = weightsGrads.find('img.train-weights-gradients-'+layer+'-'+i);
                    if (img.length === 0) {
                        weightsGrads.append('<img class="train-weights-gradients-'+layer+'-'+i+'" height="32" style="border: solid 1px #fff">');
                        img = weightsGrads.find('img.train-weights-gradients-'+layer+'-'+i);
                    }

                    img[0].src = 'data:image/png;base64,' + event.data.WeightsGradients[i];
                }
            }

            if (event.data.OutputImages) {
                var outputImages = box.find(".output-images")
                let layer = event.data.Index
                for (var i = 0; i < event.data.OutputImages.length; i++) {
                    let img = outputImages.find('img.train-output-gradients-'+layer+'-'+i);
                    if (img.length === 0) {
                        outputImages.append('<img class="train-output-gradients-'+layer+'-'+i+'" height="32" style="border: solid 1px #fff">');
                        img = outputImages.find('img.train-output-gradients-'+layer+'-'+i);
                    }

                    img[0].src = 'data:image/png;base64,' + event.data.OutputImages[i];
                }
            }

            if (event.data.WeightsImages) {
                var weightsImages = box.find(".weights-images")
                let layer = event.data.Index
                for (var i = 0; i < event.data.WeightsImages.length; i++) {
                    var img = weightsImages.find('img.train-weights-gradients-'+layer+'-'+i);
                    if (img.length === 0) {
                        weightsImages.append('<img class="train-weights-gradients-'+layer+'-'+i+'" height="32" style="border: solid 1px #fff">');
                        img = weightsImages.find('img.train-weights-gradients-'+layer+'-'+i);
                    }

                    img[0].src = 'data:image/png;base64,' + event.data.WeightsImages[i];
                }
            }

            if (event.data.Weights) {
                var weightsBox = box.find(".weights")
                let layer = event.data.Index

                for (var i = 0; i < event.data.Weights.length; i++) {
                    var plotBox = weightsBox.find('div.weights-plot-'+layer+'-'+i);
                    if (plotBox.length === 0) {
                        weightsBox.append('<div class="weights-plot-'+layer+'-'+i+'" style="width: 200px; height: 150px; float: left"></div>');
                    }

                    var min = 0.0
                    var max = 0.0

                    for (var k = 0; k < event.data.Weights[i].length; k++) {
                        var v = event.data.Weights[i][k];
                        if (k === 0 || v < min) {
                            min = v
                        }
                        if (k === 0 || v > max) {
                            max = v
                        }
                    }

                    var cnt = 25;
                    var step = (max-min)/cnt;
                    var d = [];

                    for (var k = 0; k < cnt+1; k++) {
                        d[k] = 0;
                    }

                    for (var k = 0; k < event.data.Weights[i].length; k++) {
                        var v = event.data.Weights[i][k];
                        var idx = Math.floor((v-min)/step);
                        d[idx]++
                    }

                    var dd = [];
                    for (var k = 0; k < d.length; k++) {
                        dd.push([min+k*step, d[k]])
                    }

                    $.plot('div.weights-plot-'+layer+'-'+i, [ dd ], {
                        series: {
                            bars: {
                                show: true,
                                barWidth: 0.01,
                                align: "center"
                            }
                        },
                        xaxis: {
                            // mode: "categories",
                            // showTicks: false,
                            // gridLines: false
                        }
                    });
                    //
                    // $.plot('div.weights-plot-'+layer+'-'+i, [
                    //     {
                    //         data: dd,
                    //         lines: { show: true, steps: true, fill: true },
                    //         yaxis: {
                    //             show: false
                    //         }
                    //     }
                    // ]);
                    //
                    // plot = $.plot('div.weights-plot-'+layer+'-'+i, [dd], {
                    //     series: {
                    //         shadowSize: 0
                    //     },
                    //     yaxis: {
                    //         show: true
                    //     },
                    //     xaxis: {
                    //         show: true
                    //     }
                    // });
                    //
                    // plot.draw();
                }
            }
        });


        var trainLossIndex = 0;
        var trainLossData = [];
        var checkLossData = [];

        self.wsListener.Bind(self.eventCode("train-loss"), self.wsRecipient, function (event) {
            trainLossData.push([trainLossIndex, event.data.loss]);
            checkLossData.push([trainLossIndex, event.data.testLoss]);
            trainLossIndex++;

            if (trainLossData.length > 25) {
                trainLossData.shift();
                checkLossData.shift();
            }

            let maxv = 0;
            let minv = 0;

            for (var i = 0; i < trainLossData.length; i++) {
                if (maxv < trainLossData[i][1]) {
                    maxv = trainLossData[i][1];
                }
                if (maxv < checkLossData[i][1]) {
                    maxv = checkLossData[i][1];
                }
                if (minv > trainLossData[i][1] || i === 1) {
                    minv = trainLossData[i][1];
                }
                if (minv > checkLossData[i][1] || i === 1) {
                    minv = checkLossData[i][1];
                }
            }

            plot = $.plot("#" + self.netboxid + " .loss-plot", [trainLossData, checkLossData], {
                series: {
                    shadowSize: 0
                },
                yaxis: {
                    show: true,
                    // min: minv * 0.0,
                    min: 0.0,
                    max: maxv * 1.2,
                },
                xaxis: {
                    show: true,
                    min: trainLossData[0][0],
                }
            });

            plot.draw();
        });


        var durationCounter = 0
        var duration = [];

        self.wsListener.Bind(self.eventCode("train-duration"), self.wsRecipient, function (event) {
            duration.push([durationCounter, event.data]);
            durationCounter++;

            if (duration.length > 50) {
                duration.shift();
            }

            let maxv = 0;
            let minv = 0;

            for (var i = 0; i < duration.length; i++) {
                if (maxv < duration[i][1]) {
                    maxv = duration[i][1];
                }
                if (minv > duration[i][1] || i === 1) {
                    minv = duration[i][1];
                }
            }

            plot = $.plot("#" + self.netboxid + " .duration-plot", [duration], {
                series: {
                    shadowSize: 0
                },
                yaxis: {
                    show: true,
                    min: minv * 0.8,
                    max: maxv * 1.2,
                },
                xaxis: {
                    show: true,
                    min: duration[0][0],
                }
            });

            plot.draw();
        });


        var trainSuccessRate = [];
        var checkSuccessRate = [];
        var successI = 0;

        self.wsListener.Bind(self.eventCode("success-rate"), self.wsRecipient, function (event) {
            trainSuccessRate.push([successI, event.data.train]);
            checkSuccessRate.push([successI, event.data.check]);
            successI++;

            if (trainSuccessRate.length > 100) {
                trainSuccessRate.shift();
                checkSuccessRate.shift();
            }

            plot = $.plot("#" + self.netboxid + " .success-rate-plot", [trainSuccessRate, checkSuccessRate], {
                series: {
                    shadowSize: 0
                },
                yaxis: {
                    show: true,
                    min: 0,
                    max: 110,
                },
                xaxis: {
                    show: true,
                    min: trainSuccessRate[0][0],
                }
            });

            plot.draw();
        });
    };

    // self.CreateNet = function () {
    //     self.wsListener.wsClient.sendJSON({
    //         code: self.eventCode("create"),
    //         data: {}
    //     });
    // };
    //
    // self.TrainNet = function () {
    //     self.wsListener.wsClient.sendJSON({
    //         code: self.eventCode("train"),
    //         data: {}
    //     });
    // };

    return this;
}