import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging
logging.set_verbosity_error()



def BERT(res,filedata):
    matchedfiles = []
    for file in res:
        match = re.search("\.py$", file)
        if match:
            matchedfiles.append(file)
    dic = {}
    for i in matchedfiles:
        content = filedata[i]
        functions = content.split("def ")
        functions.pop(0)
        lens = len(functions)
        c = 0
        for y in functions:
            if c == lens:
                break
            name = functions[c].split("(")
            dic[name[0]] = y
            c += 1
        imports = content.split("import ")
        finalimports = []
        length = len(imports)
        cc = 0
        for y in imports:
            sc = 0
            for s in y:
                if s=="#":
                    break
                elif s=="\n":
                    finalimports.append(y[:sc])
                    break
                elif s==" ":
                    finalimports.append(y[:sc])
                    break
                elif s==",":
                    continue
                sc += 1
        packages = content.split("from ")
        lpl = len(packages)
        c = 1
        for y in packages:
            if c == lpl:
                break
            name = packages[c].split(" ")
            c += 1

    print("\n\n\n")
    print('Functions in the repository: ',end="\n\n")
    print('Using RoBERTa  - ',end="\n\n")

    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

    def test_entailment(text1, text2):
        batch = tokenizer(text1, text2, return_tensors='pt').to(model.device)
        with torch.no_grad():
            proba = torch.softmax(model(**batch).logits, -1)
        return proba.cpu().numpy()[0, model.config.label2id['ENTAILMENT']]

    def test_equivalence(text1, text2):
        return test_entailment(text1, text2) * test_entailment(text2, text1)

    matchscores3 = {}
    for key in dic:
        #matchscores3[key] = test_equivalence(key,session['keyword'])
        matchscores3[key] = test_equivalence(key,'keee')
    sorted_dict3 = dict(reversed(list(sorted(matchscores3.items(), key = lambda kv3: kv3[1]))))
    order = []
    for item in sorted_dict3:
        print(item+', Match Score: ',sorted_dict3[item],end="\n")
        order.append(item)
    print(filedata)
    print(sorted_dict3)
    return order

app = Flask(__name__)
app.secret_key = 'xyz'
app.config['UPLOAD_FOLDER'] = './uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt','arw','md','html','py','json'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("file[]")
    session['keyword'] = request.form['keyword']
    filenames = []
    filedata = {}
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            filenames.append(filename)
            with app.open_resource('./uploads/'+filename) as f:
                contents = f.read()
            filedata[file.filename] = contents
    session['filenames'] = filenames
    order = BERT(filenames,filedata)
    return render_template('upload.html', filenames=order)


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True,threaded=True)