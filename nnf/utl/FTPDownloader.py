# -*- coding: utf-8 -*-
"""
.. module:: FTPDownloader
   :platform: Unix, Windows
   :synopsis: Represent FTPDownloader class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from ftplib import FTP
import os

# Local Imports


class FTPDownloader(object):
    """File downloader via FTP"""

    def __init__(self, host, user, passwd):
        self.ftp = FTP(host=host, user=user, passwd=passwd)

    def download(self, local_root, remote_root):

        # Create a local directory if it does not exist
        if not os.path.isdir(local_root):
            os.makedirs(local_root)

        # Change the local working directory to the local_root
        os.chdir(local_root)

        # Change the remote working directory to remote_root
        self.ftp.cwd(remote_root)

        # Fetch the filename list (including folder names)
        filenames = self.ftp.nlst()

        # Iterate each file in the filenames array
        for filename in filenames:

            # Ignore the following
            if filename == '.' or filename == '..':
                continue

            # Copy the file
            if self.__is_file(filename):

                full_local_path = os.path.join(local_root, filename)
                full_remote_path = remote_root + "/" + filename
                print('copy: ' + full_remote_path + ' to ' + full_local_path)

                file = open(filename, 'wb')
                self.ftp.retrbinary('RETR '+ filename, file.write)
                file.close()

            else:
                new_local_root = os.path.join(local_root, filename)
                new_remote_root = remote_root + "/" + filename
                self.download(new_local_root, new_remote_root)

                # Change the local working directory back to the original local_root
                os.chdir(local_root)

                # Change the remote working directory back to original remote_root
                self.ftp.cwd(remote_root)

    def __is_file(self, filename):
        current = self.ftp.pwd()

        try:
            self.ftp.cwd(filename)
        except:
            self.ftp.cwd(current)
            return True

        self.ftp.cwd(current)
        return False




