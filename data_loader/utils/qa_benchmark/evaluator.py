import boto3
import time
import json
import tqdm
import math
from openai import OpenAI, NotGiven
import os
import re
from enum import Enum
from typing import Dict, Any, Union, List, Tuple, Iterable, Callable
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import concurrent.futures
from functools import partial
from datetime import datetime

load_dotenv()


def read_json(file_path: str) -> dict:
    """Read the JSON file from the given file path."""
    with open(file_path, "r") as file:
        return json.load(file)


class Answer(BaseModel):
    entity: str = Field(..., description="An answer which is a chemical entity.")


class AreSimilar(BaseModel):
    are_the_same: bool = Field(
        ..., description="Whether the two entities are the same."
    )


JSON_ENFORCE = (
    "Please return only the raw JSON string (no markdown) that strictly conforms to the following JSON schema, with no additional text: {json_schema}"
    "Example: {example}"
)


class Provider(str, Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    NVIDIA = "nvidia"


class ModelRegistry:
    """Registry to manage provider-model relationships and benchmarking state."""

    PROVIDER_MODELS = {
        Provider.OPENAI: [
            "gpt-4o",
            # "gpt-4o-mini",
            # "o1",
            # "o1-mini",
            # "o3-mini",
        ],
        Provider.BEDROCK: [
            # "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            # "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.meta.llama3-3-70b-instruct-v1:0",
            # "anthropic.claude-3-5-sonnet-20240620-v1:0",
            # "mistral.mistral-large-2402-v1:0",
            # "us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning",
            "us.deepseek.r1-v1:0-reasoning",
        ],
        Provider.NVIDIA: [
            # "deepseek-ai/deepseek-r1",
        ],
    }

    def __init__(self):
        self.completed_benchmarks = set()

    def get_models_for_provider(self, provider: Provider) -> List[str]:
        """Get all models supported by a specific provider."""
        return self.PROVIDER_MODELS.get(provider, [])

    def is_valid_model(self, provider: Provider, model: str) -> bool:
        """Check if a model is valid for a given provider."""
        return model in self.PROVIDER_MODELS.get(provider, [])

    def get_all_provider_model_combinations(self) -> List[Tuple[Provider, str]]:
        """Get all valid provider-model combinations."""
        combinations = []
        for provider in Provider:
            for model in self.get_models_for_provider(provider):
                combinations.append((provider, model))
        return combinations

    def mark_as_completed(self, provider: Provider, model: str):
        """Mark a provider-model combination as completed."""
        self.completed_benchmarks.add((provider, model))

    def is_completed(self, provider: Provider, model: str) -> bool:
        """Check if a provider-model combination has been completed."""
        return (provider, model) in self.completed_benchmarks

    def load_completed_benchmarks(self, responses_dir: str):
        """Load already completed benchmarks from response files."""
        self.completed_benchmarks = set()
        if not os.path.exists(responses_dir):
            return

        for filename in os.listdir(responses_dir):
            if filename.startswith("responses_") and filename.endswith(".json"):
                parts = (
                    filename.replace("responses_", "").replace(".json", "").split("_")
                )
                if len(parts) >= 2:
                    try:
                        # Convert string back to Provider enum
                        provider_str = parts[0]
                        provider = next(
                            (p for p in Provider if p.value == provider_str), None
                        )
                        if not provider:
                            continue

                        model = parts[1]
                        if (
                            "__" in model
                        ):  # Handle cases like 'deepseek-ai__deepseek-r1'
                            model = model.replace("__", "/")
                        self.mark_as_completed(provider, model)
                    except (ValueError, KeyError):
                        continue


class StructuredLLM:
    def __init__(
        self,
        provider: Union[Provider, str],
        model_id: str,
        output_format: BaseModel,
        temperature: float = 0.2,
        max_completion_tokens: int = 8192,
    ):
        self.provider = (
            provider if isinstance(provider, Provider) else Provider(provider)
        )
        self.model_id = model_id
        self.output_format = output_format
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.is_reasoning = False
        self.thinking_params = None

        model_registry = ModelRegistry()
        if not model_registry.is_valid_model(self.provider, self.model_id):
            raise ValueError(
                f"Model '{self.model_id}' is not supported by provider '{self.provider}'"
            )

        if "reasoning" in self.model_id:
            self.model_id = self.model_id.replace("-reasoning", "")
            self.is_reasoning = True
            if "claude-3-7" in self.model_id:
                self.temperature = 1.0
                self.thinking_params = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": self.max_completion_tokens - 256,
                    }
                }
            else:
                self.temperature = 0.6

        if self.provider in [Provider.OPENAI, Provider.NVIDIA]:
            self.api_key = self._get_api_key()
        self.client = self._initialize_client()
        if self.provider == Provider.BEDROCK:
            self.bedrock_llm = self._get_bedrock_llm()

        if self.provider == Provider.OPENAI and self.model_id in [
            "o1",
            "o1-preview",
            "o1-mini",
            "o3-mini",
        ]:
            self.temperature = NotGiven()

    def _get_api_key(self) -> str:
        """Get API key from environment variables based on the provider."""
        key_mapping = {
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.NVIDIA: "NVIDIA_API_KEY",
        }

        if self.provider in key_mapping:
            env_var = key_mapping[self.provider]
            api_key = os.environ.get(env_var)
            if not api_key:
                raise ValueError(f"{env_var} environment variable is not set")
            return api_key

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == Provider.OPENAI:
            return OpenAI(api_key=self.api_key)
        elif self.provider == Provider.NVIDIA:
            return OpenAI(
                base_url="https://integrate.api.nvidia.com/v1", api_key=self.api_key
            )
        elif self.provider == Provider.BEDROCK:
            session = boto3.session.Session()
            configured_region = session.region_name
            return boto3.client("bedrock-runtime", region_name=configured_region)
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    def _parse_json_from_text(self, text_to_parse: str) -> BaseModel:
        """Extract and parse JSON from text string."""
        try:
            parsed_json = JsonOutputParser().invoke(text_to_parse)
            parsed_output = self.output_format.model_validate(parsed_json)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            parsed_output = self._generate_empty_output()

        return parsed_output

    def _process_unstructured_stream(self, stream: Iterable[AIMessage]):
        responses = []
        usage_metadata = {}
        text = ""
        reason = ""
        for x in stream:
            if x.content:
                for content in x.content:
                    if "type" in content and content["type"] == "text":
                        text += content["text"]
                    if "type" in content and content["type"] == "reasoning_content":
                        reasoning_content = content["reasoning_content"]
                        if "text" in reasoning_content:
                            reason += reasoning_content["text"]
            if x.response_metadata:
                if "metrics" in x.response_metadata:
                    latency = x.response_metadata["metrics"]["latencyMs"]
                    latency = latency[0] if isinstance(latency, list) else latency
            if x.usage_metadata:
                usage_metadata["input_tokens"] = x.usage_metadata["input_tokens"]
                usage_metadata["output_tokens"] = x.usage_metadata["output_tokens"]
            responses.append(x)
        parsed_output = self._parse_json_from_text(text)
        return {"raw": responses}, reason, parsed_output, latency, usage_metadata

    def _get_bedrock_llm(self):
        """Get the Bedrock LLM model based on the model name and temperature."""
        llm = ChatBedrockConverse(
            client=self.client,
            model_id=self.model_id,
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            additional_model_request_fields=self.thinking_params,
        )
        llm = (
            llm.with_structured_output(self.output_format, include_raw=True)
            if not self.is_reasoning
            else llm
        )
        return llm

    def _call_bedrock(self, messages: list[dict]) -> Dict[str, Any]:
        """Call the Bedrock LLM model with the given message."""
        reason = None
        if self.model_id in [
            "us.deepseek.r1-v1:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.meta.llama3-3-70b-instruct-v1:0",
        ]:
            streams = self.bedrock_llm.stream(messages)
            if self.is_reasoning:
                response, reason, parsed_output, latency, usage_metadata = (
                    self._process_unstructured_stream(streams)
                )
            else:
                response = next(streams)
                latency = response["raw"].response_metadata["metrics"]["latencyMs"]
                latency = latency[0] if isinstance(latency, list) else latency
                usage_metadata = response["raw"].usage_metadata
                output_text = (
                    response["raw"].content[0]["input"]
                    if isinstance(response["raw"].content, list)
                    else response["raw"].content
                )
                parsed_output = self._parse_json_from_text(output_text)

        else:
            response = self.bedrock_llm.invoke(messages)
            usage_metadata = response["raw"].usage_metadata
            parsed_output = response["parsed"]
            latency = response["raw"].response_metadata["metrics"]["latencyMs"][0]

        output = {
            "raw_response": response["raw"],
            "parsed_output": parsed_output,
            "date": datetime.now(),
            "latency": latency,
            "input_tokens": usage_metadata["input_tokens"],
            "output_tokens": usage_metadata["output_tokens"],
        }
        if reason:
            output["reasoning"] = reason
        return output

    def _call_openai(self, messages: str) -> Dict[str, Any]:
        """Call the OpenAI API with the given messages."""
        now = time.time()
        if self.model_id == "o1-mini":
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
            )
            parsed_output = self._parse_json_from_text(
                response.choices[0].message.content
            )
        else:
            response = self.client.beta.chat.completions.parse(
                model=self.model_id,
                messages=messages,
                response_format=self.output_format,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
            )
            parsed_output = response.choices[0].message.parsed
        elapsed_ms = (time.time() - now) * 1000
        output = {
            "raw_response": response,
            "parsed_output": parsed_output,
            "date": datetime.now(),
            "latency": elapsed_ms,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens,
        }
        return output

    def _call_nvidia(self, messages: str) -> Dict[str, Any]:
        now = time.time()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.6,
            top_p=0.7,
            max_tokens=self.max_completion_tokens,
        )
        elapsed_ms = (time.time() - now) * 1000
        response_text = response.choices[0].message.content

        reasoning_tokens = None
        text_to_parse = response_text

        think_pattern = r"<think>(.*?)</think>(.*)"
        think_match = re.search(think_pattern, response_text, re.DOTALL)

        if think_match:
            reasoning_tokens = think_match.group(1).strip()
            text_to_parse = think_match.group(2).strip()

        parsed_output = self._parse_json_from_text(text_to_parse)

        output = {
            "raw_response": response,
            "parsed_output": parsed_output,
            "reasoning_tokens": reasoning_tokens,
            "date": datetime.now(),
            "latency": elapsed_ms,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "reasoning": reasoning_tokens,
        }
        return output

    def _generate_empty_output(self):
        """Create an empty instance of the output format."""
        field_types = self.output_format.__annotations__
        fields = {}
        for field_name, field_type in field_types.items():
            if field_type == str:
                fields[field_name] = ""
            elif field_type == bool:
                fields[field_name] = False
            elif field_type == int:
                fields[field_name] = 0
            elif field_type == float:
                fields[field_name] = 0.0
            else:
                fields[field_name] = None
        return self.output_format(**fields)

    def __call__(self, prompt: str) -> Dict[str, Any]:
        """Evaluate the given messages using the appropriate LLM model."""
        # Todo: Add support for dynamic example, instead of hardcoding
        json_schema = JSON_ENFORCE.format(
            json_schema=self.output_format.model_json_schema(),
            example='{"entity": "Aspirin"}',
        )
        prompt = f"{prompt}\n{json_schema}"

        if self.provider == Provider.BEDROCK:
            messages = [{"role": "user", "content": [{"text": prompt}]}]
            return self._call_bedrock(messages)
        elif self.provider == Provider.OPENAI:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            return self._call_openai(messages)
        elif self.provider == Provider.NVIDIA:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            return self._call_nvidia(messages)


class Evaluate:
    def __init__(
        self,
        qa_llm: StructuredLLM,
        records: list[dict],
        responses_save_path: str = None,
        verifier_provider: Provider = Provider.OPENAI,
        verifier_model: str = "gpt-4o",
        num_workers: int = 16,
    ):
        self.qa_llm = qa_llm
        self.verifier_llm = StructuredLLM(
            provider=verifier_provider,
            model_id=verifier_model,
            output_format=AreSimilar,
        )
        self.records = records
        self.responses_save_path = responses_save_path
        self.num_workers = num_workers

        records_per_worker = len(records) / num_workers
        self.batch_size = max(1, math.ceil(records_per_worker))

        self.qa_llm_params = {
            "provider": qa_llm.provider,
            "model_id": qa_llm.model_id,
            "output_format": qa_llm.output_format,
            "temperature": qa_llm.temperature,
            "max_completion_tokens": qa_llm.max_completion_tokens,
        }

        if qa_llm.is_reasoning:
            self.qa_llm_params["model_id"] = f"{qa_llm.model_id}-reasoning"

        self.verifier_llm_params = {
            "provider": verifier_provider,
            "model_id": verifier_model,
            "output_format": AreSimilar,
        }

    def _verify_entity(
        self, expected: str, candidate: str, worker_verifier_llm=None
    ) -> bool:
        verifier_llm = worker_verifier_llm or self.verifier_llm

        if candidate.strip().lower() == expected.strip().lower():
            return True
        else:
            VERIFY_PROMPT = (
                f"Expected answer: {expected}\n"
                f"Candidate answer: {candidate}\n"
                "Are these two answers referring to the same entity? "
                "Answer with true or false."
            )
            response = verifier_llm(VERIFY_PROMPT)
            return response["parsed_output"].are_the_same

    def _create_worker_llms(self):
        """Create new LLM instances for workers to avoid thread safety issues"""
        qa_llm = StructuredLLM(**self.qa_llm_params)
        verifier_llm = StructuredLLM(**self.verifier_llm_params)
        return qa_llm, verifier_llm

    def _process_record_without_context(self, record, worker_llms=None):
        """Process a single record without context (thread-safe)"""
        qa_llm, verifier_llm = worker_llms or (self.qa_llm, self.verifier_llm)

        question = record["question"]
        expected = record["expected"]

        NO_CONTEXT_PROMPT = (
            "Provide a single entity name as an answer to the following question:\n"
            f"Question: {question}"
        )

        response = qa_llm(NO_CONTEXT_PROMPT)

        if response["parsed_output"] is not None:
            candidate = response["parsed_output"].entity
            is_correct = (
                self._verify_entity(expected, candidate, verifier_llm)
                if expected
                else False
            )
        else:
            candidate = None
            is_correct = False

        return {
            "candidate": candidate,
            "is_correct": is_correct,
            "context_used": False,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "reasoning_tokens": response.get("reasoning_tokens", 0),
            "latency": round(response.get("latency", 0), 2),
            "date": response.get("date", datetime.now()).strftime("%Y-%m-%d %H:%M"),
            "reasoning": response.get("reasoning", None),
            "raw": record,
        }

    def _process_record_with_context(self, record, worker_llms=None):
        """Process a single record with context (thread-safe)"""
        qa_llm, verifier_llm = worker_llms or (self.qa_llm, self.verifier_llm)

        question = record["question"]
        context = record["context"]
        expected = record["expected"]

        WITH_CONTEXT_PROMPT = (
            "Provide a single entity name as an answer to the following question based on the context:\n"
            f"Question: {question}\n\n"
            f"Context: {context}"
        )

        response = qa_llm(WITH_CONTEXT_PROMPT)

        if response["parsed_output"] is not None:
            candidate = response["parsed_output"].entity
            is_correct = (
                self._verify_entity(expected, candidate, verifier_llm)
                if expected
                else False
            )
        else:
            candidate = None
            is_correct = False

        return {
            "candidate": candidate,
            "is_correct": is_correct,
            "context_used": True,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "reasoning_tokens": response.get("reasoning_tokens", 0),
            "latency": round(response.get("latency", 0), 2),
            "date": response.get("date", datetime.now()).strftime("%Y-%m-%d %H:%M"),
            "reasoning": response.get("reasoning", None),
            "raw": record,
        }

    def _process_batch(self, batch, process_fn):
        """Process a batch of records with the given processing function"""
        # Create worker-specific LLM instances to avoid thread safety issues
        worker_llms = self._create_worker_llms()
        return [process_fn(record, worker_llms) for record in batch]

    def _parallel_process(self, process_fn: Callable):
        """Process all records in parallel using the provided function"""
        results = []

        batches = [
            self.records[i : i + self.batch_size]
            for i in range(0, len(self.records), self.batch_size)
        ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            batch_fn = partial(self._process_batch, process_fn=process_fn)

            futures = [executor.submit(batch_fn, batch) for batch in batches]

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(batches)
            ):
                batch_results = future.result()
                results.extend(batch_results)

        assert len(results) == len(self.records), "Some records were not processed"

        return results

    def _eval_without_context(self) -> List[Dict[str, Any]]:
        """Evaluate records without context using parallel processing"""
        return self._parallel_process(self._process_record_without_context)

    def _eval_with_context(self) -> List[Dict[str, Any]]:
        """Evaluate records with context using parallel processing"""
        return self._parallel_process(self._process_record_with_context)

    def evaluate(self):
        """
        Evaluate the records with and without context, and return results.
        If responses_save_path is provided, save the detailed results as JSON.
        """
        without_context_results = self._eval_without_context()
        with_context_results = self._eval_with_context()

        if self.responses_save_path:
            all_results = without_context_results + with_context_results
            with open(self.responses_save_path, "w") as f:
                json.dump(all_results, f, default=str, indent=2)

        model_name = f"{self.qa_llm.provider.value}-{self.qa_llm.model_id}"

        results = []

        if without_context_results:
            total = len(without_context_results)
            correct = sum(1 for r in without_context_results if r["is_correct"])
            accuracy = (correct / total) * 100 if total > 0 else 0
            avg_latency = (
                sum(r["latency"] for r in without_context_results) / total
                if total > 0
                else 0
            )
            avg_in_tokens = (
                sum(r["input_tokens"] for r in without_context_results) / total
                if total > 0
                else 0
            )
            avg_out_tokens = (
                sum(r["output_tokens"] for r in without_context_results) / total
                if total > 0
                else 0
            )
            total_in_tokens = sum(r["input_tokens"] for r in without_context_results)
            total_out_tokens = sum(r["output_tokens"] for r in without_context_results)

            results.append(
                {
                    "Model": model_name,
                    "Run": "Without Context",
                    "Accuracy (%)": round(accuracy, 2),
                    "Avg Duration (s)": round(avg_latency, 2),
                    "Avg Input Tokens": round(avg_in_tokens, 2),
                    "Avg Output Tokens": round(avg_out_tokens, 2),
                    "Total Input Tokens": total_in_tokens,
                    "Total Output Tokens": total_out_tokens,
                    "Total Samples": total,
                }
            )

        if with_context_results:
            total = len(with_context_results)
            correct = sum(1 for r in with_context_results if r["is_correct"])
            accuracy = (correct / total) * 100 if total > 0 else 0
            avg_latency = (
                sum(r["latency"] for r in with_context_results) / total
                if total > 0
                else 0
            )
            avg_in_tokens = (
                sum(r["input_tokens"] for r in with_context_results) / total
                if total > 0
                else 0
            )
            avg_out_tokens = (
                sum(r["output_tokens"] for r in with_context_results) / total
                if total > 0
                else 0
            )
            total_in_tokens = sum(r["input_tokens"] for r in with_context_results)
            total_out_tokens = sum(r["output_tokens"] for r in with_context_results)

            results.append(
                {
                    "Model": model_name,
                    "Run": "With Context",
                    "Accuracy (%)": round(accuracy, 2),
                    "Avg Duration (s)": round(avg_latency, 2),
                    "Avg Input Tokens": round(avg_in_tokens, 2),
                    "Avg Output Tokens": round(avg_out_tokens, 2),
                    "Total Input Tokens": total_in_tokens,
                    "Total Output Tokens": total_out_tokens,
                    "Total Samples": total,
                }
            )

        return pd.DataFrame(results)


class BenchmarkRunner:
    """Class to run benchmarks across multiple models and providers."""

    def __init__(
        self,
        records: list[dict],
        responses_dir: str = "responses",
        results_file: str = "results.csv",
    ):
        self.records = records
        self.responses_dir = responses_dir
        self.results_file = results_file
        self.model_registry = ModelRegistry()
        os.makedirs(responses_dir, exist_ok=True)

        self.model_registry.load_completed_benchmarks(responses_dir)

    def run_all_benchmarks(self, skip_completed: bool = True):
        """Run benchmarks for all provider-model combinations."""
        results_all = []

        for (
            provider,
            model,
        ) in self.model_registry.get_all_provider_model_combinations():
            if skip_completed and self.model_registry.is_completed(provider, model):
                print(f"Skipping already evaluated {provider.value} model: {model}")
                continue

            print(f"\nEvaluating {provider.value} model: {model}")
            try:
                structured_llm = StructuredLLM(
                    provider=provider, model_id=model, output_format=Answer
                )

                model_filename = model.replace("/", "__")
                responses_path = f"{self.responses_dir}/responses_{provider.value}_{model_filename}.json"

                evaluator = Evaluate(
                    qa_llm=structured_llm,
                    records=self.records,
                    responses_save_path=responses_path,
                )

                df_results = evaluator.evaluate()
                results_all.append(df_results)

                self.model_registry.mark_as_completed(provider, model)

            except Exception as e:
                print(f"Error evaluating {provider.value} model {model}: {e}")

        if results_all:
            combined_results = pd.concat(results_all)
            combined_results.to_csv(self.results_file, index=False)
            return combined_results

        return pd.DataFrame()


if __name__ == "__main__":
    RESPONSES_DIR = "responses"  # Directory to save intermediate responses
    RESULT_PATH = "results.csv"  # Save the final results as a CSV file
    RECORDS_PATH = "records.json"  # Input json, list of dictionaries with keys 'question', 'context', 'expected'
    os.makedirs(RESPONSES_DIR, exist_ok=True)

    records = read_json(RECORDS_PATH)

    benchmark_runner = BenchmarkRunner(
        records=records, responses_dir=RESPONSES_DIR, results_file=RESULT_PATH
    )
    now = datetime.now()
    benchmark_runner.run_all_benchmarks(skip_completed=True)
    elapsed = datetime.now() - now
    elapsed_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed.total_seconds()))
    print(f"Completed benchmarking in {elapsed_formatted}")
