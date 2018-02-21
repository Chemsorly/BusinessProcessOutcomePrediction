using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualBasic.FileIO;
using MoreLinq;

namespace LSTMdecoder
{
    class Program
    {
        static void Main(string[] args)
        {
            //open file; original c2k csv dataset
            const String InFilepath = @"D:\Desktop\Masterarbeit\c2k_data_comma.csv";
            const String OutFilepath_single = @"D:\Desktop\Masterarbeit\c2k_data_comma_lstmready.csv";
            const String OutFilepath_multi = @"D:\Desktop\Masterarbeit\c2k_data_comma_lstmready_multi.csv";

            SingleEventProcessing.CreateSingleEventLog(InFilepath, OutFilepath_single);
            MultiEventProcessing.CreateSingleEventLog(InFilepath, OutFilepath_multi);

        }
    }
}
