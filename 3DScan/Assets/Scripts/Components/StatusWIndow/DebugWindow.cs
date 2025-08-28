using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Core;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityProgressBar;

public class DebugWindow : MonoBehaviour, IProgressHandler
{
    [SerializeField] ProgressBar progressBar;
    [SerializeField] TextMeshProUGUI promptTextbox;
    private bool finished = false;

    // unity 
    private void Start()
    {
        EndStage();
    }

    private void Update()
    {

    }

    public void StartStage(float weight, string description = null)
    {
        finished = false;
        Debug.Log($"Started: {description}");

        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            gameObject.SetActive(true);
            progressBar.Value = 0;
            promptTextbox.SetText(description ?? "Starting stage\n");
        });

    }

    public void EndStage()
    {
        finished = true;
        Debug.Log($"Finished");

        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            gameObject.SetActive(true);
            progressBar.Value = 1;
            promptTextbox.SetText(promptTextbox.text + " - All tasks completed");
            
            StartCoroutine(Delay());
            IEnumerator Delay()
            {
                yield return new WaitForSeconds(3);
                if (finished)
                    gameObject.SetActive(false);
            }
        });
    }

    public void ReportProgress(float progress, string description = null)
    {
        finished = false;
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            gameObject.SetActive(true);
            progressBar.Value = progress;
            if (description != null)
                promptTextbox.SetText(description + "\n(" + (int)progress * 100 + "/" + 100 + ")");
        });
    }

    public void ReportProgress(int currentStep, int totalSteps, string description = null)
    {
        finished = false;
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            gameObject.SetActive(true);
            progressBar.Value = (float)currentStep / totalSteps;
            if (description != null)
                promptTextbox.SetText(description + "\n(" + currentStep + "/" + totalSteps + ")");
        });
    }

    public void Log(string description = null)
    {
        finished = false;
        Debug.Log(description);
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            gameObject.SetActive(true);
            promptTextbox.SetText(promptTextbox.text + "\n" + description);
        });
    }
    
    public void Fail()
    {
        finished = false;
        Debug.LogError($"Error");
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            gameObject.SetActive(true);
            progressBar.Value = 1;
            promptTextbox.SetText(promptTextbox.text + " - Failed");
        });
    }
    public void Fail(string description = null)
    {
        Debug.LogError($"Error: " + description);
        Log("Error: " + description);
    }
}
