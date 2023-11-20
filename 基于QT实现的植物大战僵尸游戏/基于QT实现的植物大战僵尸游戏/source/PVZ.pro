#-------------------------------------------------
#
# Project created by QtCreator 2018-07-28T14:45:10
#
#-------------------------------------------------

QT       += core gui \
            multimedia

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = PVZ
TEMPLATE = app


SOURCES += main.cpp\
        maindialog.cpp \
    zscene.cpp \
    zobject.cpp \
    zplant.cpp \
    zzombie.cpp \
    zflyingobject.cpp \
    zanim.cpp \
    zbonus.cpp \
    zcard.cpp

HEADERS  += maindialog.h \
    zscene.h \
    zobject.h \
    zplant.h \
    zzombie.h \
    zflyingobject.h \
    zanim.h \
    zbonus.h \
    zcard.h


RESOURCES += \
    pvz.qrc

RC_FILE = proj.rc
