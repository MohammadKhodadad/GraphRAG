import boto3
import time
import json
from openai import OpenAI, NotGiven
import os
import re
from enum import Enum
from typing import Dict, Any, Optional, Union, List, Tuple, Set
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from pydantic import BaseModel, Field
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

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


JSON_ENFORCE = "Please return only the raw JSON string (no markdown) that strictly conforms to the following JSON schema, with no additional text: {json_schema}"


class Provider(str, Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    NVIDIA = "nvidia"


class ModelRegistry:
    """Registry to manage provider-model relationships and benchmarking state."""

    PROVIDER_MODELS = {
        Provider.OPENAI: [
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
            "o3-mini",
        ],
        Provider.BEDROCK: [
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning",
            "us.meta.llama3-3-70b-instruct-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "mistral.mistral-large-2402-v1:0",
        ],
        Provider.NVIDIA: [
            "deepseek-ai/deepseek-r1",
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
        max_completion_tokens: int = 4096,
    ):
        self.provider = (
            provider if isinstance(provider, Provider) else Provider(provider)
        )
        self.model_id = model_id
        self.output_format = output_format
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.thinking_params = None

        model_registry = ModelRegistry()
        if not model_registry.is_valid_model(self.provider, self.model_id):
            raise ValueError(
                f"Model '{self.model_id}' is not supported by provider '{self.provider}'"
            )

        if "reasoning" in self.model_id:
            self.model_id = self.model_id.replace("-reasoning", "")
            self.temperature = 1
            self.thinking_params = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self.max_completion_tokens - 256,
                }
            }

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

    def _parse_json_from_text(self, text_to_parse: str):
        """Extract and parse JSON from text string."""
        json_pattern = r"{.*}"
        json_match = re.search(json_pattern, text_to_parse, re.DOTALL)

        if json_match:
            try:
                json_text = json_match.group(0).replace("'", '"')
                parsed_output = self.output_format.model_validate_json(json_text)
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                parsed_output = self._generate_empty_output()
        else:
            parsed_output = self._generate_empty_output()

        return parsed_output

    def _get_bedrock_llm(self):
        """Get the Bedrock LLM model based on the model name and temperature."""
        llm = ChatBedrockConverse(
            client=self.client,
            model_id=self.model_id,
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            additional_model_request_fields=self.thinking_params,
        )
        llm = llm.with_structured_output(self.output_format, include_raw=True)
        return llm

    def _call_bedrock(self, messages: list[dict]) -> Dict[str, Any]:
        """Call the Bedrock LLM model with the given message."""
        reason = ""
        if self.model_id in [
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.meta.llama3-3-70b-instruct-v1:0",
        ]:
            streams = self.bedrock_llm.stream(messages)
            response = next(streams)
            latency = response["raw"].response_metadata["metrics"]["latencyMs"]
            usage_metadata = response["raw"].usage_metadata
            if self.thinking_params:
                # First content is reasoning, the second is tool use
                try:
                    reason = response["raw"].content[0]["reasoning_content"]["text"]
                except Exception as e:
                    reason = None
                parsed_output = self._parse_json_from_text(
                    response["raw"].content[1]["input"]
                )
            else:
                # No thinking, only the tool use
                parsed_output = self._parse_json_from_text(
                    response["raw"].content[0]["input"]
                )

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
        prompt = f"{prompt}\n{JSON_ENFORCE.format(json_schema=self.output_format.model_json_schema())}"
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
        verifier_model: str = "o3-mini",
    ):
        self.qa_llm = qa_llm
        self.verifier_llm = StructuredLLM(
            provider=verifier_provider,
            model_id=verifier_model,
            output_format=AreSimilar,
        )
        self.records = records
        self.responses_save_path = responses_save_path

    def _verify_entity(self, expected: str, candidate: str) -> bool:
        if candidate.strip().lower() == expected.strip().lower():
            return True
        else:
            VERIFY_PROMPT = (
                f"Expected answer: {expected}\n"
                f"Candidate answer: {candidate}\n"
                "Are these two answers referring to the same entity? "
                "Answer with true or false."
            )
            response = self.verifier_llm(VERIFY_PROMPT)
            return response["parsed_output"].are_the_same

    def _eval_without_context(self) -> List[Dict[str, Any]]:
        results = []

        NO_CONTEXT_PROMPT = (
            "Provide a single entity name as an answer to the following question:\n"
            "Question: {question}"
        )

        for record in self.records:
            question = record["question"]
            expected = record["expected"]

            response = self.qa_llm(NO_CONTEXT_PROMPT.format(question=question))

            if response["parsed_output"] is not None:
                candidate = response["parsed_output"].entity
                is_correct = (
                    self._verify_entity(expected, candidate) if expected else False
                )
            else:
                candidate = None
                is_correct = False

            result = {
                "candidate": candidate,
                "is_correct": is_correct,
                "context_used": False,
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
                "reasoning_tokens": response.get("reasoning_tokens", 0),
                "latency": round(response.get("latency", 0), 2),
                "date": response.get("date", datetime.now()).strftime("%Y-%m-%d %H:%M"),
                "raw": record,
            }
            results.append(result)

        return results

    def _eval_with_context(self) -> List[Dict[str, Any]]:
        results = []

        WITH_CONTEXT_PROMPT = (
            "Provide a single entity name as an answer to the following question based on the context:\n"
            "Question: {question}\n\n"
            "Context: {context}"
        )

        for record in self.records:
            question = record["question"]
            context = record["context"]
            expected = record["expected"]

            response = self.qa_llm(
                WITH_CONTEXT_PROMPT.format(question=question, context=context)
            )

            if response["parsed_output"] is not None:
                candidate = response["parsed_output"].entity
                is_correct = (
                    self._verify_entity(expected, candidate) if expected else False
                )
            else:
                candidate = None
                is_correct = False

            result = {
                "candidate": candidate,
                "is_correct": is_correct,
                "context_used": True,
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
                "reasoning_tokens": response.get("reasoning_tokens", 0),
                "latency": round(response.get("latency", 0), 2),
                "date": response.get("date", datetime.now()).strftime("%Y-%m-%d %H:%M"),
                "raw": record,
            }
            results.append(result)

        return results

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
            # try:
            structured_llm = StructuredLLM(
                provider=provider, model_id=model, output_format=Answer
            )

            model_filename = model.replace("/", "__")
            responses_path = (
                f"{self.responses_dir}/responses_{provider.value}_{model_filename}.json"
            )

            evaluator = Evaluate(
                qa_llm=structured_llm,
                records=self.records,
                responses_save_path=responses_path,
            )

            df_results = evaluator.evaluate()
            results_all.append(df_results)

            self.model_registry.mark_as_completed(provider, model)

            # except Exception as e:
            #     print(f"Error evaluating {provider.value} model {model}: {e}")

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

    benchmark_runner.run_all_benchmarks(skip_completed=True)
