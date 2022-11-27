package views

import (
	"bytes"
	"html/template"
	"io"
	"strings"

	"github.com/pkg/errors"
)

func ParseViews(root string, paths ...string) (*Views, error) {
	templates := template.New("templates")
	for _, path := range paths {
		_, err := templates.ParseGlob(root + path)
		if err != nil {
			return nil, errors.Wrapf(err, "parse path %s%s failed with error:", root, path)
		}
	}
	return &Views{Template: templates}, nil
}

type Views struct {
	*template.Template
}

func (t *Views) RenderLayout(w io.Writer, name string, data interface{}) error {
	return t.ExecuteTemplate(w, "layout/"+strings.Trim(name, "/"), data)
}

func (t *Views) RenderView(w io.Writer, name string, data interface{}) error {
	return t.ExecuteTemplate(w, "views/"+strings.Trim(name, "/"), data)
}

func (t *Views) Render(w io.Writer, layout, view string, layoutData, viewData map[string]interface{}) error {
	buf := bytes.NewBuffer(nil)
	if err := t.RenderView(buf, view, viewData); err != nil {
		return errors.Wrap(err, "can't render view")
	}

	if layoutData == nil {
		layoutData = map[string]interface{}{}
	}

	layoutData["content"] = template.HTML(buf.String())

	buf.Reset()

	if err := t.RenderLayout(buf, layout, layoutData); err != nil {
		return errors.Wrap(err, "can't render layout")
	}

	if _, err := buf.WriteTo(w); err != nil {
		return errors.Wrap(err, "can't flush buffer")
	}

	return nil
}
