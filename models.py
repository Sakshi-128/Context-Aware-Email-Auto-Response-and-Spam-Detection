from django.db import models
from django.utils import timezone
# Create your models here.


class AdminMaster(models.Model):
    ad_id = models.AutoField(primary_key=True, unique=True)
    ad_name = models.CharField(max_length=100)
    ad_mobile = models.CharField(max_length=100)
    ad_email = models.CharField(max_length=100)
    ad_password = models.CharField(max_length=100)
    ad_role = models.CharField(max_length=100)
    ad_status = models.CharField(max_length=100, default="0")
    ad_created_by = models.CharField(max_length=100)


class Register(models.Model):
    rg_id = models.AutoField(primary_key=True, unique=True)
    rg_name = models.CharField(max_length=100)
    rg_mobile = models.CharField(max_length=100)
    rg_email = models.CharField(max_length=100)
    rg_password = models.CharField(max_length=100)
    rg_address = models.CharField(max_length=100, default="")
    rg_secret_key = models.CharField(max_length=100, default="")
    rg_status = models.CharField(max_length=100, default="0")


class Contact(models.Model):
    co_id = models.AutoField(primary_key=True, unique=True)
    co_name = models.CharField(max_length=100)
    co_mobile = models.CharField(max_length=100)
    co_email = models.CharField(max_length=100)
    co_subject = models.CharField(max_length=100)
    co_message = models.CharField(max_length=100)
    co_status = models.CharField(max_length=100)

class Emails(models.Model):
    em_id = models.AutoField(primary_key=True, unique=True)
    em_name = models.CharField(max_length=100)
    em_to = models.CharField(max_length=100)
    em_subject = models.CharField(max_length=100)
    em_message = models.TextField()
    em_reply = models.TextField(default="")
    em_spam = models.TextField(default="NO")
    em_type = models.CharField(max_length=100, default="SENT")
    em_status = models.CharField(max_length=100, default="NoTrash")
    em_created_at = models.DateTimeField(default=timezone.now)

class SpamEmails(models.Model):
    sm_id = models.AutoField(primary_key=True, unique=True)
    sm_email = models.CharField(max_length=100)
