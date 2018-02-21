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
    class MultiEventProcessing
    {
        private static readonly string[] Columns = new[]
        {
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
            "V", "W", "X", "Y", "Z"
        };

        public static void CreateSingleEventLog(String pInFilepath, String pOutFilepath)
        {
            using (TextFieldParser parser = new TextFieldParser(pInFilepath))
            {
                //internal data structure
                //first level list: CaseID (sequence)
                //second level list: Activity (timesteps)
                //String values for [0] = ActivityID, [1] = duration, [2] = time since last event
                List<Sequence> sequences = new List<Sequence>();

                //https://stackoverflow.com/questions/3507498/reading-csv-files-using-c-sharp
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");
                bool firstline = true;
                while (!parser.EndOfData)
                {
                    //Processing row; each row is a sequence of events
                    string[] fields = parser.ReadFields();

                    //check if id is valid (aka not empty); ignore first line (header)
                    if (firstline || String.IsNullOrEmpty(fields[0]))
                    {
                        firstline = false;
                        continue;
                    }

                    //in c2k dataset, we target for row as CaseID, 
                    //Column3 (i1_rcs_e) as Activity1 with duration(minutes) as value
                    //Column5 (i1_dep_1_e) as Activity2 with duration(minutes) as value
                    //Column8 (i1_rcf_1_e) as Activity3 with duration(minutes) as value
                    //Column11 (i1_dep_2_e) as Activity2 with duration(minutes) as value
                    //Column14 (i1_rcf_2_e) as Activity3 with duration(minutes) as value
                    //Column17 (i1_dep_3_e) as Activity2 with duration(minutes) as value
                    //Column20 (i1_rcf_3_e) as Activity3 with duration(minutes) as value
                    //Column23 (i1_dlv_e) as Activity4 with duration(minutes) as value

                    //Column27 (i2_rcs_e) as Activity5 with duration(minutes) as value
                    //Column29 (i2_dep_1_e) as Activity6 with duration(minutes) as value
                    //Column32 (i2_rcf_1_e) as Activity7 with duration(minutes) as value
                    //Column35 (i2_dep_2_e) as Activity6 with duration(minutes) as value
                    //Column38 (i2_rcf_2_e) as Activity7 with duration(minutes) as value
                    //Column41 (i2_dep_3_e) as Activity6 with duration(minutes) as value
                    //Column44 (i2_rcf_3_e) as Activity7 with duration(minutes) as value
                    //Column47 (i2_dlv_e) as Activity8 with duration(minutes) as value

                    //Column51 (i3_rcs_e) as Activity9 with duration(minutes) as value
                    //Column53 (i3_dep_1_e) as Activity10 with duration(minutes) as value
                    //Column56 (i3_rcf_1_e) as Activity11 with duration(minutes) as value
                    //Column59 (i3_dep_2_e) as Activity10 with duration(minutes) as value
                    //Column62 (i3_rcf_2_e) as Activity11 with duration(minutes) as value
                    //Column65 (i3_dep_3_e) as Activity10 with duration(minutes) as value
                    //Column68 (i3_rcf_3_e) as Activity11 with duration(minutes) as value
                    //Column71 (i3_dlv_e) as Activity12 with duration(minutes) as value

                    //Column75 (i4_rcs_e) as Activity13 with duration(minutes) as value
                    //Column77 (i4_dep_1_e) as Activity14 with duration(minutes) as value
                    //Column80 (i4_rcf_1_e) as Activity15 with duration(minutes) as value
                    //Column83 (i4_dep_2_e) as Activity14 with duration(minutes) as value
                    //Column86 (i4_rcf_2_e) as Activity15 with duration(minutes) as value
                    //Column89 (i4_dep_3_e) as Activity14 with duration(minutes) as value
                    //Column92 (i4_rcf_3_e) as Activity15 with duration(minutes) as value
                    //Column95 (i4_dlv_e) as Activity16 with duration(minutes) as value
                    //planned is always effective -1


                    List<int> columnnumbers = new List<int>() { 3, 5, 8, 11, 14, 17, 20, 23, 27, 29, 32, 35, 38, 41, 44, 47, 51, 53, 56, 59, 62, 65, 68, 71, 75, 77, 80, 83, 86, 89, 92, 95 };
                    List<int> leg1Activitynumbers = new List<int>() { 1, 2, 3, 4 };
                    List<int> leg2Activitynumbers = new List<int>() { 5, 6, 7, 8 };
                    List<int> leg3Activitynumbers = new List<int>() { 9, 10, 11, 12 };
                    List<int> leg4Activitynumbers = new List<int>() { 13, 14, 15, 16 };

                    List<Event> leg1Events = new List<Event>();
                    List<Event> leg2Events = new List<Event>();
                    List<Event> leg3Events = new List<Event>();
                    List<Event> leg4Events = new List<Event>();

                    //iterate through columns and find activity
                    foreach (var columnnumber in columnnumbers)
                    {
                        if (fields[columnnumber] != "?")
                        {
                            //check for activity
                            int activityid = -1;
                            if (columnnumber == 3)
                                activityid = 1;
                            else if (columnnumber == 5 || columnnumber == 11 || columnnumber == 17)
                                activityid = 2;
                            else if (columnnumber == 8 || columnnumber == 14 || columnnumber == 20)
                                activityid = 3;
                            else if (columnnumber == 23)
                                activityid = 4;
                            else if (columnnumber == 27)
                                activityid = 5;
                            else if (columnnumber == 29 || columnnumber == 35 || columnnumber == 41)
                                activityid = 6;
                            else if (columnnumber == 32 || columnnumber == 38 || columnnumber == 44)
                                activityid = 7;
                            else if (columnnumber == 47)
                                activityid = 8;
                            else if (columnnumber == 51)
                                activityid = 9;
                            else if (columnnumber == 53 || columnnumber == 59 || columnnumber == 65)
                                activityid = 10;
                            else if (columnnumber == 56 || columnnumber == 62 || columnnumber == 68)
                                activityid = 11;
                            else if (columnnumber == 71)
                                activityid = 12;
                            else if (columnnumber == 75)
                                activityid = 13;
                            else if (columnnumber == 77 || columnnumber == 83 || columnnumber == 89)
                                activityid = 14;
                            else if (columnnumber == 80 || columnnumber == 86 || columnnumber == 92)
                                activityid = 15;
                            else if (columnnumber == 95)
                                activityid = 16;
                            else
                                continue;

                            //create event and add to leg collection
                            Event legevent = new Event
                            {
                                ProcessID = int.Parse(fields[0]),
                                ActivityID = activityid,
                                Duration = int.Parse(fields[columnnumber]),
                                PlannedDuration = int.Parse(fields[columnnumber - 1])
                            };

                            //leg 1
                            if (leg1Activitynumbers.Contains(activityid))
                                leg1Events.Add(legevent);
                            //leg 2
                            else if (leg2Activitynumbers.Contains(activityid))
                                leg2Events.Add(legevent);
                            //leg 3
                            else if (leg3Activitynumbers.Contains(activityid))
                                leg3Events.Add(legevent);
                            //leg 4
                            else if (leg4Activitynumbers.Contains(activityid))
                                leg4Events.Add(legevent);
                            else
                                throw new Exception("unknown activityid");
                        }
                    }
                    //leg events are sorted chronologically 
                    //calculate timestamp as total time since start
                    int totaltime = 0;
                    int plannedtotaltime = 0;
                    leg1Events.ForEach(t =>
                    {
                        totaltime += t.Duration;
                        plannedtotaltime += t.PlannedDuration;
                        t.Timestamp = totaltime;
                        t.PlannedTimestamp = plannedtotaltime;
                    });
                    totaltime = 0;
                    plannedtotaltime = 0;
                    leg2Events.ForEach(t =>
                    {
                        totaltime += t.Duration;
                        plannedtotaltime += t.PlannedDuration;
                        t.Timestamp = totaltime;
                        t.PlannedTimestamp = plannedtotaltime;
                    });
                    totaltime = 0;
                    plannedtotaltime = 0;
                    leg3Events.ForEach(t =>
                    {
                        totaltime += t.Duration;
                        plannedtotaltime += t.PlannedDuration;
                        t.Timestamp = totaltime;
                        t.PlannedTimestamp = plannedtotaltime;
                    });

                    //leg 4 starts as soon leg 1-3 finished. therefor leg 4 starts at the latest timestamp of leg 1-3
                    totaltime = leg1Events.Concat(leg2Events).Concat(leg3Events).Max(t => t.Timestamp);
                    plannedtotaltime = leg1Events.Concat(leg2Events).Concat(leg3Events).Max(t => t.PlannedTimestamp);
                    leg4Events.ForEach(t =>
                    {
                        totaltime += t.Duration;
                        plannedtotaltime += t.PlannedDuration;
                        t.Timestamp = totaltime;
                        t.PlannedTimestamp = plannedtotaltime;
                    });

                    //merge and sort by Timestamp
                    List<Event> AllLegs = leg1Events.Concat(leg2Events).Concat(leg3Events).Concat(leg4Events).OrderBy(t => t.Timestamp).ToList();

                    //calculate time since last event
                    for (int i = 0; i < AllLegs.Count; i++)
                    {
                        //special case 0
                        if (i == 0)
                            AllLegs[i].TimeSinceLastEvent = AllLegs[i].Timestamp;
                        else
                            AllLegs[i].TimeSinceLastEvent = AllLegs[i].Timestamp - AllLegs[i - 1].Timestamp;
                    }

                    //transform to activity design (instead of having an ID about the finished process step we supply a string for all currently running events)
                    //go through event list of process, find all events that run at the same time and transform to string
                    foreach(var eEvent in AllLegs)
                    {
                        //find events whose start time > current events start time && start time < current events end time
                        //AllLegs.Where(t => t.StartTimestamp >= eEvent.StartTimestamp && t.StartTimestamp < eEvent.Timestamp).ToList();
                        var otherevents = new List<Event>();
                        if (leg1Events.Any())
                        {
                            var leg1event = leg1Events
                                .Where(t => t.StartTimestamp < eEvent.Timestamp &&
                                            t.Timestamp >= eEvent.Timestamp);
                            if (leg1event.Any())
                                otherevents.Add(leg1event.MaxBy(t => t.StartTimestamp));
                        }
                        if (leg2Events.Any())
                        {
                            var leg2event = leg2Events
                                .Where(t => t.StartTimestamp < eEvent.Timestamp &&
                                            t.Timestamp >= eEvent.Timestamp);
                            if (leg2event.Any())
                                otherevents.Add(leg2event.MaxBy(t => t.StartTimestamp));
                        }
                        if (leg3Events.Any())
                        {
                            var leg3event = leg3Events
                                .Where(t => t.StartTimestamp < eEvent.Timestamp &&
                                            t.Timestamp >= eEvent.Timestamp);
                            if (leg3event.Any())
                                otherevents.Add(leg3event.MaxBy(t => t.StartTimestamp));
                        }
                        if (leg4Events.Any())
                        {
                            var leg4event = leg4Events
                                .Where(t => t.StartTimestamp < eEvent.Timestamp &&
                                            t.Timestamp >= eEvent.Timestamp);
                            if (leg4event.Any())
                                otherevents.Add(leg4event.MaxBy(t => t.StartTimestamp));
                        }

                        //transform into string
                        eEvent.ActivityString = String.Empty;
                        foreach (var concurrentevent in otherevents)
                            eEvent.ActivityString += Columns[concurrentevent.ActivityID - 1];
                    }

                    //add to list
                    sequences.Add(new Sequence() { EventsInSequence = AllLegs });
                }
                //target data structure
                //csv with
                //"CaseID,ActivityID,Duration,TimeSinceLastEvent,Timestamp"
                //CaseID ordered but not unique; 
                //ActivityID ordere chronologically by activity; 
                //duration is the actual duration; 
                //timesincelastevent the time since the previous event (multiple legs!)
                //timestamp time since event start
                List<String> exportrows = new List<string>();
                exportrows.Add("CaseID,ActivityString,Duration,TimeSinceLastEvent,Timestamp,PlannedDuration,PlannedTimestamp,InstanceID,EndTimestamp,StartTime,FinishedActivity,PlannedEndTimestamp");
                for (int caseid = 0; caseid < sequences.Count; caseid++)
                {
                    foreach (var activity in sequences[caseid].EventsInSequence)
                    {
                        exportrows.Add($"{caseid},{activity.ActivityString},{activity.Duration},{activity.TimeSinceLastEvent},{activity.Timestamp},{activity.PlannedDuration},{activity.PlannedTimestamp},{activity.ProcessID},{sequences[caseid].EndTimestamp},{activity.StartTimestamp},{Columns[activity.ActivityID - 1]},{sequences[caseid].PlannedEndTimestamp}");
                    }
                }

                ////export as csv to match LSTM input examples
                File.WriteAllLines(pOutFilepath, exportrows);
            }
        }

        class Event
        {
            /// <summary>
            /// activityid classification by process model
            /// </summary>
            public int ActivityID { get; set; }

            public String ActivityString { get; set; }

            /// <summary>
            /// duration of the event in minutes
            /// </summary>
            public int Duration { get; set; }

            /// <summary>
            /// the planned (allowed) duration for the event
            /// </summary>
            public int PlannedDuration { get; set; }

            /// <summary>
            /// the timestamp at event end (durations added up)
            /// </summary>
            public int Timestamp { get; set; }

            /// <summary>
            /// the planned timestamp for the activity
            /// </summary>
            public int PlannedTimestamp { get; set; }

            /// <summary>
            /// time since last event over all events
            /// </summary>
            public int TimeSinceLastEvent { get; set; }

            /// <summary>
            /// the process instance ID
            /// </summary>
            public int ProcessID { get; set; }

            public int StartTimestamp => Timestamp - Duration;
        }

        class Sequence
        {
            public List<Event> EventsInSequence { get; set; }

            public int EndTimestamp => EventsInSequence.Max(t => t.Timestamp);
            public int PlannedEndTimestamp => EventsInSequence.Max(t => t.PlannedTimestamp);
        }
    }
}
