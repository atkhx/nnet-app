package images

import (
	"bytes"
	"image"
	"image/color"
	"image/png"

	"github.com/atkhx/nnet/data"
	"github.com/pkg/errors"
)

func CreateImagesFromDataMatrixesWithAverageValues(data *data.Data) ([][]byte, error) {
	mw, mh, mz := data.Dims[0], data.Dims[1], data.Dims[2]
	res := [][]byte{}

	for z := 0; z < mz; z++ {
		img := image.NewGray(image.Rect(0, 0, mw, mh))
		min, max := data.GetMinMaxValues(z*mw*mh, (z+1)*mw*mh)
		max -= min

		for y := 0; y < mh; y++ {
			for x := 0; x < mw; x++ {
				c := data.Data[z*mw*mh+y*mw+x] - min
				img.Set(x, y, color.Gray{Y: byte(255.0 * c / max)})
			}
		}

		buf := bytes.NewBuffer(nil)
		err := png.Encode(buf, img)

		if err != nil {
			return nil, err
		}

		res = append(res, buf.Bytes())
	}
	return res, nil
}

func CreateImageFromDataWithAverageValues(data *data.Data) ([]byte, error) {
	var w, h, d int
	data.ExtractDimensions(&w, &h, &d)

	min, max := data.GetMinMaxValues(0, len(data.Data))
	max -= min

	if d == 1 {
		img := image.NewGray(image.Rect(0, 0, w, h))

		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				img.Set(x, y, color.Gray{Y: byte(255.0 * (data.Data[y*w+x] - min) / max)})
			}
		}

		buf := bytes.NewBuffer(nil)
		err := png.Encode(buf, img)

		if err != nil {
			return nil, err
		}

		return buf.Bytes(), nil
	} else if d == 3 {
		img := image.NewRGBA(image.Rect(0, 0, w, h))

		imageSQ := w * h
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				img.Set(x, y, color.RGBA{
					R: byte(255 * (data.Data[y*w+x] - min) / max),
					G: byte(255 * (data.Data[imageSQ+y*w+x] - min) / max),
					B: byte(255 * (data.Data[imageSQ+imageSQ+y*w+x] - min) / max),
					A: 255,
				})
			}
		}

		buf := bytes.NewBuffer(nil)
		err := png.Encode(buf, img)

		if err != nil {
			return nil, err
		}

		return buf.Bytes(), nil
	} else if d%3 == 0 { // experiment
		WW := w * d / 3

		img := image.NewRGBA(image.Rect(0, 0, WW, h))
		imageSQ := w * h

		for i := 0; i < d/3; i++ {
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					img.Set(x+i*w, y, color.RGBA{
						R: byte(255 * (data.Data[(i*w*h*3)+y*w+x] - min) / max),
						G: byte(255 * (data.Data[(i*w*h*3)+imageSQ+y*h+x] - min) / max),
						B: byte(255 * (data.Data[(i*w*h*3)+imageSQ+imageSQ+y*w+x] - min) / max),
						A: 255,
					})
				}
			}

		}

		buf := bytes.NewBuffer(nil)
		err := png.Encode(buf, img)

		if err != nil {
			return nil, err
		}

		return buf.Bytes(), nil
	} else {
		return nil, errors.New("data depth not equals 1 or 3")
	}
}

func CreateImageFromData(data *data.Data) ([]byte, error) {
	var w, h, d int
	data.ExtractDimensions(&w, &h, &d)

	if d == 1 {
		img := image.NewGray(image.Rect(0, 0, w, h))

		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				img.Set(x, y, color.Gray{Y: byte(255.0 * data.Data[y*w+x])})
			}
		}

		buf := bytes.NewBuffer(nil)
		err := png.Encode(buf, img)

		if err != nil {
			return nil, err
		}

		return buf.Bytes(), nil
	} else if d == 3 {
		img := image.NewRGBA(image.Rect(0, 0, w, h))

		imageSQ := w * h
		for y := 0; y < w; y++ {
			for x := 0; x < h; x++ {
				img.Set(x, y, color.RGBA{
					R: byte(255 * data.Data[y*w+x]),
					G: byte(255 * data.Data[imageSQ+y*h+x]),
					B: byte(255 * data.Data[imageSQ+imageSQ+y*w+x]),
					A: 255,
				})
			}
		}

		buf := bytes.NewBuffer(nil)
		err := png.Encode(buf, img)

		if err != nil {
			return nil, err
		}

		return buf.Bytes(), nil
	} else {
		return nil, errors.New("data depth not equals 1 or 3")
	}
}
