from django import forms

class RunMLForm(forms.Form):

    uid = forms.CharField(widget=forms.TextInput(attrs={"class": "form-control"}), required=True)
    pdb = forms.CharField(widget=forms.Textarea(attrs={"class": "form-control"}), required=False)
    email = forms.EmailField(widget=forms.EmailInput(attrs={"class": "form-control"}), required=True)
    
