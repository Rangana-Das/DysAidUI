#django

from django.shortcuts import render,redirect
from .forms import UserRegistrationForm

def home(request):
    return render(request, 'home.html')
    
def login(request):
    return render(request,'login.html')
    
def reg(request):
    if request.method == 'POST':
        form=UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('main-login')
    else:
        form = UserRegistrationForm()
    return render(request,'reg.html',{'form':form})

