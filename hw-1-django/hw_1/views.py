from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import onnxruntime
import numpy as np
from PIL import Image

imageClassList = {'0': 'Бледная поганка', '1': 'Мухомор', '2': 'Подосиновик'}  #Сюда указать классы
imageCriteriaList = {'0': 'Белый цвет, наличие выраженной вольвы', '1': 'Красноватый цвет, толстомясистая шляпка, наличие вкрапинок', '2': 'Наличие оранжево-коричневого цвета, ножка не совсем белая'}
imageDescription = {'0': 'Подосиновик - съедобный гриб с крупной коричневой шляпкой, короткой толстой ножкой и белым трубчатым слоем под шляпкой. Часто встречается в лиственных лесах.',
                    '1': 'Мухомор - яркий гриб с красной шляпкой, усеянной белыми пятнами, и белой ножкой с кольцом. Часто встречается в хвойных и лиственных лесах.',
                    '2': 'Бледная поганка - гриб с белой или серовато-белой шляпкой, ножкой без кольца и пластинчатым слоем под шляпкой. Он часто встречается в лесах и на лугах.'}
imageTasty = {'0': 'Подосиновики считаются съедобными и широко используются в кулинарии.',
                    '1': 'Многие виды мухоморов считаются ядовитыми и могут вызывать серьезное отравление. Не рекомендуется употреблять без должной экспертизы.',
                    '2': 'Бледная поганка является ядовитым грибом и не подходит для употребления в пищу.'}
imageLatin = {'0': 'Leccinum spp',
                    '1': 'Amanita spp',
                    '2': 'Chlorophyllum spp'}

def scoreImagePage(request):
    return render(request, 'scorepage.html')

def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save('images/'+fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    modelName = request.POST.get('modelName')
    scorePrediction, criteria, desc, tasty, latin = predictImageData(modelName, '.'+filePathName)
    context = {'scorePrediction': scorePrediction, 'imagePath': filePathName, 'criteria': criteria, 'desc': desc, 'tasty': tasty, 'latin': latin}
    return render(request, 'scorepage.html', context)

def predictImageData(modelName, filePath):
    img = Image.open(filePath).convert("RGB")
    img = np.asarray(img.resize((32, 32), Image.ANTIALIAS))
    sess = onnxruntime.InferenceSession(r'/home/artyom/VUZ_Learning/DLIU/hw-1-django/media/models/mushrooms_CNN_MobileNet.onnx') #<-Здесь требуется указать свой путь к модели
    outputOFModel = np.argmax(sess.run(None, {'input': np.asarray([img]).astype(np.float32)}))
    score = imageClassList[str(outputOFModel)]
    criteria =  imageCriteriaList[str(outputOFModel)]
    desc =  imageDescription[str(outputOFModel)]
    tasty =  imageTasty[str(outputOFModel)]
    latin =  imageLatin[str(outputOFModel)]
    return score, criteria, desc, tasty, latin