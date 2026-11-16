"""Public receipt output and preview boundaries; no host observation on reads."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
class CliTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.env=dict(os.environ,SPARK_STATE=str(self.root/'state.json'))
    def cli(self,*args):
        return subprocess.run([str(ROOT/'tools/cluster'),*args],env=self.env,capture_output=True,text=True,timeout=10)
    def test_saved_legacy_queued_failure_and_ready_keep_public_exit_codes(self):
        receipt=self.root/'receipt.json'
        for status,code in (('queued',3),('failed',1),('ready',0)):
            receipt.write_text(json.dumps({'status':status,'target':'studio-4','stage':'complete' if status=='ready' else 'queued'}))
            result=self.cli('profile','status','--operation',str(receipt))
            self.assertEqual(result.returncode,code,result.stderr)
            view=json.loads(result.stdout)
            self.assertEqual(view['status'],status);self.assertEqual(view['ownership'],'unverified')
            self.assertTrue(view['historical'])
        self.assertFalse((self.root/'state.json').exists())
    def test_profile_preview_uses_saved_profile_without_mutation(self):
        for name in ('studio-4','jobs','jobs-studio','jobs-studio-pair'):
            result=self.cli('profile','apply',name)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['effect'],'preview')
        self.assertFalse((self.root/'state.json').exists())
    def test_bad_name_and_missing_operation_are_errors(self):
        self.assertNotEqual(self.cli('profile','apply','../invalid').returncode,0)
        self.assertEqual(self.cli('profile','resume','unknown','--yes').returncode,1)

if __name__=='__main__':unittest.main()
