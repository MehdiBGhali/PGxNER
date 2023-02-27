from fastNLP.core.callback import Callback
from fastNLP import DataSet, Tester
import fitlog
from copy import deepcopy
import os
import re
from datetime import datetime
import time


class FitlogCallback(Callback):
    r"""
    This callback can write Loss and Progress into Fitlog; if Trainer has DEV data, it will automatically 
    write the results of the DEV into the log; One (or more) TEST dataset is tested (can only be used when 
    Trainer has DEV), and each time on the Evaluate on the DEV, it will be verified on these datasets.
    And write the verification results into Fitlog.The results of these data sets are reported according 
    to the best results on DEV, that is, if DEV is the best in the third EPOCH, then the best, then the best, then
    The result of the data set recorded in Fitlog is the result of the third EPOCH.
    """

    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=1, log_exception=True,
                 raise_threshold=0, better_dev_eval=True, eval_begin_epoch=-1):
        r"""

        : Param ~ FastNLP.DataSet, Dict [~ Fastnlp.dataset] data: Pass into the DataSet object, and use 
        Metric in multiple traineers to verify the data.If needed
        For multiple datasets, please pass it through DICT. DICT's key will be passed to Fitlog as a name 
        corresponding to DataSet.The name of the data starts with 'data'.
        : Param ~ FastNLP.TESTER, DICT [~ FastNLP.TETETETETER] TESTER: Tester object, will be called 
        during on_valid_end.The name of the result of TESTER starts with 'TESTER'
        : Param int log_loss_every: How many STEP records Loss once (record the average value of these BATCH LOSS). 
        If the dataset is large, it is recommended to set the value.
            Big, otherwise it will lead to huge log files.The default is 0, that is, don't record Loss.
        : Param Int Verbose: Whether to print the results of Evaluation in the terminal, 0 will not be printed.
        : Param Bool Log_exception: Fitlog whether the Exception information that occurs occurred
        : Param Float Raise_threshold: If the Metric value is lower than this, it will be raise Exception
        : Param Bool Better_dev_eval: Only when DEV gets better results, you can do Evaluate
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        self.raise_threshold = raise_threshold
        self.eval_begin_epoch = eval_begin_epoch

        assert isinstance(log_loss_every, int) and log_loss_every>=0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0
        self.best_test_metric_sofar = 0
        self.best_test_sofar = None
        self.best_test_epoch = 0
        self.best_dev_test = None
        self.best_dev_epoch = 0
        self.better_dev_eval = better_dev_eval

    def on_train_begin(self):
        print("fitlog callback is go")
        if (len(self.datasets) > 0 or len(self.testers) > 0) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.trainer.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.kwargs.get('test_use_tqdm', self.trainer.use_tqdm),
                                sampler=self.trainer.kwargs.get('test_sampler', None))
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)

        if self.trainer.save_path is not None:
            model_name = "best_" + "_".join([self.model.__class__.__name__, self.trainer.metric_key, self.trainer.start_time])
            fitlog.add_other(name='model_name', value=model_name)

    def on_epoch_begin(self):
        if self.eval_begin_epoch>0 and self.epoch>self.eval_begin_epoch:
            self.trainer.validate_every = -1

    def on_backward_begin(self, loss):
        if self._log_loss_every >0:
            self._avg_loss += loss.item()
            if self.step %self._log_loss_every==0:
                fitlog.add_loss(self._avg_loss /self._log_loss_every *self.update_every, name='loss', step=self.step, epoch=self.epoch)
                self._avg_loss = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        indicator, indicator_val = _check_eval_results(eval_result, metric_key=metric_key)
        if indicator_val < self.raise_threshold:
            raise RuntimeError("The program has been running off.")

        if len(self.testers) > 0:
            do_eval = True
            if self.better_dev_eval:
                if not better_result:
                    do_eval = False
            if do_eval:
                for idx, (key, tester) in enumerate(self.testers.items()):
                    try:
                        eval_result = tester.test()
                        if self.verbose != 0:
                            self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                            self.pbar.write(tester._format_eval_results(eval_result))
                        fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                        if idx == 0:
                            indicator, indicator_val = _check_eval_results(eval_result, metric_key=self.trainer.metric_key)
                            if indicator_val>self.best_test_metric_sofar:
                                self.best_test_metric_sofar = indicator_val
                                self.best_test_epoch = self.epoch
                                self.best_test_sofar = eval_result

                        if better_result:
                            self.best_dev_test = eval_result
                            self.best_dev_epoch = self.epoch
                            fitlog.add_best_metric(eval_result, name=key)
                    except Exception as e:
                        self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                        raise e

    def on_train_end(self):
        if self.best_test_sofar:
            line1 = "Best test performance(may not correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_test_sofar, self.best_test_epoch)
            self.logger.info(line1)
            fitlog.add_to_line(line1)
        if self.best_dev_test:
            line2 = "Best test performance(correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_dev_test, self.best_dev_epoch)
            self.logger.info(line2)
            fitlog.add_to_line(line2)
        fitlog.finish()

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')


def _check_eval_results(metrics, metric_key=None):
    # metrics: tester返回的结果
    # metric_key: 一个用来做筛选的指标，来自Trainer的初始化
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]  # 取第一个metric

        if metric_key is None:
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        else:
            # metric_key is set
            if metric_key not in metric_dict:
                raise RuntimeError(f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))
    return indicator, indicator_val


from fastNLP import WarmupCallback as FWarmupCallback
import math
class WarmupCallback(FWarmupCallback):

    def __init__(self, warmup=0.1, schedule='constant'):
        """

        : Param Int, Float Warmup: If Warmup is int, before the Step, Learning Rate changes based on Schedule; 
        if Warmup is Float, it is Float, and it is Float, and it
            If 0.1, the top 10%of STEP is to adjust the Learning Rate according to the Schedule strategy.
        : Param Str schedule: In which way is adjusted.
            Linear: Former Warmup's Step rose to the specified Learning Rate (obtained from the Optimizer in Trainer),
            and then the STEP of Warmup dropped to 0; The Step of Warmup before constant rose to the specified 
            Learning Rate, and the Step behind kept Learning Rate.
        """
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        elif schedule == 'inverse_square':
            self.get_lr = self._get_inverse_square_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_inverse_square_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return max((math.sqrt(progress) - 1.) / (math.sqrt(self.warmup) - 1.), 0.)


class OutputIndiceCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_batch_begin(self, batch_x, batch_y, indices):
        self.indices = indices

    def on_exception(self, exception):
        print(self.indices)

# Custom callbacks
class CustomFitlogCallback(Callback):
    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=1, log_exception=True, summary = None,
                 raise_threshold=0, better_dev_eval=True, eval_begin_epoch=-1, history_dir = None):
        r"""

        : Param ~ FastNLP.DataSet, Dict [~ Fastnlp.dataset] data: Pass into the DataSet object, and use 
        Metric in multiple traineers to verify the data.If needed
        For multiple datasets, please pass it through DICT. DICT's key will be passed to Fitlog as a name 
        corresponding to DataSet.The name of the data starts with 'data'.
        : Param ~ FastNLP.TESTER, DICT [~ FastNLP.TETETETETER] TESTER: Tester object, will be called 
        during on_valid_end.The name of the result of TESTER starts with 'TESTER'
        : Param int log_loss_every: How many STEP records Loss once (record the average value of these BATCH LOSS). 
        If the dataset is large, it is recommended to set the value.
            Big, otherwise it will lead to huge log files.The default is 0, that is, don't record Loss.
        : Param Int Verbose: Whether to print the results of Evaluation in the terminal, 0 will not be printed.
        : Param Bool Log_exception: Fitlog whether the Exception information that occurs occurred
        : Param Float Raise_threshold: If the Metric value is lower than this, it will be raise Exception
        : Param Bool Better_dev_eval: Only when DEV gets better results, you can do Evaluate
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        self.raise_threshold = raise_threshold
        self.eval_begin_epoch = eval_begin_epoch
        self.summary = summary

        assert isinstance(log_loss_every, int) and log_loss_every>=0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0
        self.best_test_metric_sofar = 0
        self.best_test_sofar = None
        self.best_test_epoch = 0
        self.best_dev_test = None
        self.best_dev_epoch = 0
        self.better_dev_eval = better_dev_eval
        self.history_dir = history_dir

    def on_train_begin(self):
        self.start_time = time.time()
        print("logging training to fitlog")
        if (len(self.datasets) > 0 or len(self.testers) > 0) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.trainer.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.kwargs.get('test_use_tqdm', self.trainer.use_tqdm),
                                sampler=self.trainer.kwargs.get('test_sampler', None))
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)

        if self.trainer.save_path is not None:
            model_name = "best_" + "_".join([self.model.__class__.__name__, self.trainer.metric_key, self.trainer.start_time])
            fitlog.add_other(name='model_name', value=model_name)
        
        if self.history_dir is not None:
            print('creating history dir')
            subdir_indexes = [re.search(r'\d+',f) for f in os.listdir(self.history_dir)]
            valid_indexes = [0]+[int(subidx[0]) for subidx in subdir_indexes if subidx is not None]
            print(valid_indexes)
            self.history_dir = os.path.join(self.history_dir, f"historic_{max(valid_indexes)+1}")
            if not os.path.exists(self.history_dir) : 
                os.makedirs(self.history_dir)
            with open(os.path.join(self.history_dir,"summary"), 'w') as f :
                f.write(f"Start time : {datetime.now()} \n") 
                f.write(f"\n{self.summary} \n")

    def on_epoch_begin(self):
        if self.eval_begin_epoch>0 and self.epoch>self.eval_begin_epoch:
            self.trainer.validate_every = -1

    def on_backward_begin(self, loss):
        if self._log_loss_every >0:
            self._avg_loss += loss.item()
            if self.step %self._log_loss_every==0:
                fitlog.add_loss(self._avg_loss /self._log_loss_every *self.update_every, name='loss', step=self.step, epoch=self.epoch)
                with open(os.path.join(self.history_dir,'train_loss'),'a+') as f :
                    f.write(f"{self._avg_loss /self._log_loss_every *self.update_every} \n")
                self._avg_loss = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        with open(os.path.join(self.history_dir,'valid_scores'),'a+') as f :
            f.write(f"{eval_result} \n")
        indicator, indicator_val = _check_eval_results(eval_result, metric_key=metric_key)
        if indicator_val < self.raise_threshold:
            raise RuntimeError("The program has been running off.")

        if len(self.testers) > 0:
            do_eval = True
            if self.better_dev_eval:
                if not better_result:
                    do_eval = False
            if do_eval:
                for idx, (key, tester) in enumerate(self.testers.items()):
                    try:
                        eval_result = tester.test()
                        if self.verbose != 0:
                            self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                            self.pbar.write(tester._format_eval_results(eval_result))
                        fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                        with open(os.path.join(self.history_dir,'test_scores'),'a+') as f:
                            f.write(f"{eval_result} \n")
                        if idx == 0:
                            indicator, indicator_val = _check_eval_results(eval_result, metric_key=self.trainer.metric_key)
                            if indicator_val>self.best_test_metric_sofar:
                                self.best_test_metric_sofar = indicator_val
                                self.best_test_epoch = self.epoch
                                self.best_test_sofar = eval_result

                        if better_result:
                            self.best_dev_test = eval_result
                            self.best_dev_epoch = self.epoch
                            fitlog.add_best_metric(eval_result, name=key)
                    except Exception as e:
                        self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                        raise e

    def on_train_end(self):
        if self.best_test_sofar:
            line1 = "Best test performance(may not correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_test_sofar, self.best_test_epoch)
            self.logger.info(line1)
            fitlog.add_to_line(line1)
        if self.best_dev_test:
            line2 = "Best test performance(correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_dev_test, self.best_dev_epoch)
            self.logger.info(line2)
            fitlog.add_to_line(line2)
        with open(os.path.join(self.history_dir,"summary"), 'a+') as f :
                f.write(f"\nEnd time : {datetime.now()} \n")
                f.write(f"\nTrained in : {timedelta(seconds = (time.time() - self.start_time))} \n")
                f.write(f"""\nBest test performance :{self.best_test_sofar} \n achieved at Epoch:{self.best_test_epoch} \n""")
                f.write(f"""\nTest performance corresponding to best validation :{self.best_test_sofar} 
                achieved at Epoch:{self.best_test_epoch} \n""")
        fitlog.finish() 

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')