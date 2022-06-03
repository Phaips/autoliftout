import os.path
import sys

from PyQt5.QtCore import pyqtProperty, pyqtSignal, QObject, QTextCodec, QUrl
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtWidgets import QApplication

# pip install PyQtWebEngine

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class Document(QObject):
    textChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.m_text = ""

    def get_text(self):
        return self.m_text

    def set_text(self, text):
        if self.m_text == text:
            return
        self.m_text = text
        self.textChanged.emit(self.m_text)

    text = pyqtProperty(str, fget=get_text, fset=set_text, notify=textChanged)


class DownloadManager(QObject):
    finished = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._manager = QNetworkAccessManager()
        self.manager.finished.connect(self.handle_finished)

    @property
    def manager(self):
        return self._manager

    def start_download(self, url):
        self.manager.get(QNetworkRequest(url))

    def handle_finished(self, reply):
        if reply.error() != QNetworkReply.NoError:
            print("error: ", reply.errorString())
            return
        codec = QTextCodec.codecForName("UTF-8")
        raw_data = codec.toUnicode(reply.readAll())
        self.finished.emit(raw_data)

def main():

    app = QApplication(sys.argv)

    filename = os.path.join(CURRENT_DIR, "index.html")

    document = Document()
    download_manager = DownloadManager()

    channel = QWebChannel()
    channel.registerObject("content", document)

    # remote file
    # markdown_url = QUrl.fromUserInput(
    #     "https://raw.githubusercontent.com/eyllanesc/stackoverflow/master/README.md"
    # )
    # local file
    markdown_url = QUrl.fromUserInput(r"C:\Users\Admin\Github\autoliftout\liftout\gui\README.md")

    download_manager.finished.connect(document.set_text)
    download_manager.start_download(markdown_url)

    view = QWebEngineView()
    view.page().setWebChannel(channel)
    url = QUrl.fromLocalFile(filename)
    view.load(url)
    view.resize(640, 480)
    view.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()